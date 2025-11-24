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
    // 仅 Root (Node 0) 加载 Embedding
    if (myNodeIndex == 0) {
        executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    }
    return nBytes;
}

NnSize NnLocalWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    // 所有节点都加载完整的权重 (e.g., Norms)
    executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    return nBytes;
}

NnSize NnLocalWeightLoader::loadRowMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                                      std::function<NnRowMatmulSliceUneven(NnUint)> slicer, NnByte *weight) {
    // 1. 计算本节点的切片
    NnRowMatmulSliceUneven slice = slicer(myNodeIndex);
    NnUint offset = expertIndex * slice.sliceSize.nBytes;

    // 2. 分配临时缓冲区
    allocate(slice.sliceSize.nBytes);

    // 3. 从完整权重中切分出属于我的部分
    splitRowMatmulWeightUneven(&slice, weight, temp);

    // 4. 加载
    executor->loadWeight(opName, opIndex, offset, slice.sliceSize.nBytes, temp);

    return slice.size.nBytes;
}

NnSize NnLocalWeightLoader::loadColMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                                      std::function<NnColMatmulSliceUneven(NnUint)> slicer, NnByte *weight) {
    // 1. 计算本节点的切片
    NnColMatmulSliceUneven slice = slicer(myNodeIndex);
    NnUint offset = expertIndex * slice.sliceSize.nBytes;

    // 2. 分配
    allocate(slice.sliceSize.nBytes);

    // 3. 切分
    splitColMatmulWeightUneven(&slice, weight, temp);

    // 4. 加载
    executor->loadWeight(opName, opIndex, offset, slice.sliceSize.nBytes, temp);

    return slice.size.nBytes;
}