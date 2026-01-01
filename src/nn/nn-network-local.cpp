#include "nn-network-local.hpp"
#include "nn-executor.hpp" // 需要 NnExecutor 的完整定义 (loadWeight)
#include "nn-core.hpp"     // 需要 split...Uneven 函数
#include <cstring>

NnLocalWeightLoader::NnLocalWeightLoader(NnExecutor *executor, NnUint nodeIndex) {
    this->executor = executor;
    this->myNodeIndex = nodeIndex;
    this->tempSize = 0;
    this->temp = nullptr;
}

NnLocalWeightLoader::~NnLocalWeightLoader() {
    if (tempSize > 0) delete[] temp;
}

void NnLocalWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0) delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnLocalWeightLoader::finish() {
    // 本地加载不需要网络握手
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
        temp = nullptr;
    }
}

NnSize NnLocalWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    // [Fix] Allow any node in the first stage to load embedding, not just Node 0
    // printf("DEBUG: loadRoot op=%s node=%d size=%lu\n", opName, myNodeIndex, nBytes);
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnLocalWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    // 所有节点都加载完整的权重 (e.g., Norms)
    // printf("DEBUG: loadAll op=%s node=%d size=%lu\n", opName, myNodeIndex, nBytes);
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnLocalWeightLoader::loadRowMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                                      std::function<NnRowMatmulSliceUneven(NnUint)> slicer, NnByte *weight) {
    // 1. 获取本节点的切片信息
    NnRowMatmulSliceUneven slice = slicer(myNodeIndex);
    
    // offset: 设备内存中的偏移 (用于 MoE 的 expert 偏移，非 MoE 通常为 0)
    NnUint deviceOffset = expertIndex * slice.sliceSize.nBytes;

    // [优化] Row Parallel (按行/Head 切分) 的数据在文件中是连续存储的。
    // 我们不需要 allocate temp 和 memcpy，直接计算文件内的偏移量即可。
    
    // 2. 计算本节点数据在文件中的起始偏移量
    // 计算 Block 大小 (例如 Q40 是 32 个 float 一组)
    NnSize blockSize = getBlockSize(slice.type);
    NnSize blockBytes = getBytes(slice.type, blockSize);

    // slice.n 是完整的输入维度 (Input Dim / Width)
    // 校验对齐
    if (slice.n % blockSize != 0) {
         throw std::runtime_error("RowMatmul input dim not aligned to block size");
    }

    // 计算“一行”的字节数 (Stride)
    NnSize bytesPerRow = (slice.n / blockSize) * blockBytes;

    // slice.inStart 是本节点的起始行号
    NnSize fileByteOffset = slice.inStart * bytesPerRow;

    // 3. 直接加载 (Zero-Copy)
    // weight 是当前 Tensor 的全局起始位置，weight + fileByteOffset 是本节点数据的起始位置
    // printf("DEBUG: loadRowMatmulSlicesUneven op=%s node=%d offset=%lu size=%lu\n", opName, myNodeIndex, fileByteOffset, slice.sliceSize.nBytes);
    executor->loadWeight(opName, opIndex, deviceOffset, slice.sliceSize.nBytes, weight + fileByteOffset);

    // 4. 关键：返回 Tensor 的【全局大小】，让主循环的 b 指针正确跳过整个 Tensor
    return slice.size.nBytes;
}

NnSize NnLocalWeightLoader::loadColMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                                      std::function<NnColMatmulSliceUneven(NnUint)> slicer, NnByte *weight) {
    // 1. 获取本节点的切片信息
    NnColMatmulSliceUneven slice = slicer(myNodeIndex);
    NnUint deviceOffset = expertIndex * slice.sliceSize.nBytes;

    // 2. Col Parallel (按列切分) 数据在文件中是跨步的 (Strided)。
    // 也就是数据是 [行0...][行1...]，我们需要取每一行的第 x 到 y 列。
    // 这在内存中不连续，必须分配 buffer 进行重组。
    allocate(slice.sliceSize.nBytes);

    // 3. 使用之前的辅助函数，将散落的列数据收集到 temp 缓冲区
    // splitColMatmulWeightUneven 内部已经处理了复杂的 Strided Copy 逻辑
    splitColMatmulWeightUneven(&slice, weight, temp);

    // 4. 加载重组后的连续数据
    executor->loadWeight(opName, opIndex, deviceOffset, slice.sliceSize.nBytes, temp);

    // 5. 关键：返回 Tensor 的【全局大小】
    return slice.size.nBytes;
}