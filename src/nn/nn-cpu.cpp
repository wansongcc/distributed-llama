#include "nn-cpu.hpp"
#include "nn-cpu-ops.hpp"
#include "nn-core.hpp"
#include <cassert>
#include <iostream> 
#include <cstring>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define DEBUG_CPU_OP_QUANTS false

#define BUFFER_ALIGNMENT 64

static NnByte *allocAlignedBuffer(NnSize size) {
    NnByte *buffer;
#ifdef _WIN32
    buffer = (NnByte *)_aligned_malloc(size, BUFFER_ALIGNMENT);
    if (buffer == NULL)
        throw std::runtime_error("_aligned_malloc failed");
#else
    if (posix_memalign((void **)&buffer, BUFFER_ALIGNMENT, size) != 0)
        throw std::runtime_error("posix_memalign failed");
    mlock(buffer, size);
#endif
    return buffer;
}

static void releaseAlignedBuffer(NnByte *buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif
}

static NnUint getSplitTotal(const NnDimSplit* split, NnUint nNodes) {
    if (!split || !split->lengths) return 0;
    NnUint sum = 0;
    for(NnUint i=0; i<nNodes; ++i) sum += split->lengths[i];
    return sum;
}


NnCpuDevice::NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *partitionPlan) {
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;
    this->partitionPlan = partitionPlan;

    printCpuInstructionSet();

    nBuffers = nodeConfig->nBuffers;
    buffers = new NnByte *[nBuffers];
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        NnBufferConfig *config = &nodeConfig->buffers[bufferIndex];
        NnByte *buffer = allocAlignedBuffer(config->size.nBytes);
        buffers[bufferIndex] = buffer;
    }

    bufferFlags = new NnByte[nBuffers];
    std::memset(bufferFlags, 0, nBuffers * sizeof(NnByte));
}

NnCpuDevice::~NnCpuDevice() {
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        releaseAlignedBuffer(buffers[bufferIndex]);
    }
    delete[] buffers;
    delete[] bufferFlags;
}

NnUint NnCpuDevice::maxNThreads() {
    return std::thread::hardware_concurrency();
}

NnDeviceSegment *NnCpuDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    assert(segmentConfig->nOps > 0);

    std::vector<NnOpQuantType> opQuants(segmentConfig->nOps);
    std::vector<NnCpuOpForward> opForwardLocal(segmentConfig->nOps);
    std::vector<NnSize3D> inputSizes(segmentConfig->nOps);
    std::vector<NnSize3D> outputSizes(segmentConfig->nOps);

    std::vector<std::vector<NnByte *>> inputsPtr(segmentConfig->nOps);
    std::vector<std::vector<NnByte *>> outputsPtr(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize;
        NnSize3D outputSize;
        inputsPtr[opIndex] = resolvePointer(&inputSize, &opConfig->input);
        outputsPtr[opIndex] = resolvePointer(&outputSize, &opConfig->output);

        // [Patch Start] Logits Pipe 尺寸修正补丁
        // 在非均匀切分模式下，resolvePointer 可能会根据 Pipe 的总大小计算出一个“均匀”的 Output Slice。
        // 但 inputSize (来自本地 Buffer) 是真实的“非均匀”大小。
        // 如果它们不匹配，且是 OP_CAST (常用于输出到 Pipe)，我们信任 Input 的大小。
        if (opConfig->code == OP_CAST && 
            opConfig->output.type == PNTR_BATCHED_SLICE && 
            inputSize.x != outputSize.x) {
            
            // 重新计算 3D 尺寸，保持 type, z, y 不变，仅更新 x (及其衍生的 nBytes)
            outputSize = size3D(outputSize.floatType, outputSize.z, outputSize.y, inputSize.x);
        }
        // [Patch End]

        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType);
#if DEBUG_CPU_OP_QUANTS
            printf("%20s %2d: %s\n", opConfig->name, opConfig->index, opQuantTypeToString(opQuant));
#endif
        NnCpuOpForward forward = getCpuOpForward(opConfig->code, opQuant);
        if (forward == nullptr) {
            throw std::invalid_argument(
                std::string("Unsupported CPU op code: ") + opCodeToString(opConfig->code) + 
                ", quant: " + opQuantTypeToString(opQuant) +
                ", op name: " + opConfig->name);
        }
        inputSizes[opIndex] = inputSize;
        outputSizes[opIndex] = outputSize;
        opQuants[opIndex] = opQuant;
        opForwardLocal[opIndex] = forward;
    }

    NnCpuOpForward *opForward = new NnCpuOpForward[segmentConfig->nOps];
    NnCpuOpContext *opContexts = new NnCpuOpContext[segmentConfig->nOps];

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnCpuOpContext *opContext = &opContexts[opIndex];
        NnCpuOpForwardInit opInit = getCpuOpForwardInit(opConfig->code, opQuants[opIndex]);
        opContext->name = opConfig->name;
        opContext->opConfig = opConfig->config;
        opContext->weightSize = opConfig->weightSize;
        opContext->nBatches = netConfig->nBatches;
        opContext->pipes = netExecution->pipes;
        opContext->pipeConfigs = netConfig->pipes;
        opContext->buffers = buffers;
        opContext->bufferConfigs = nodeConfig->buffers;
        opContext->bufferFlags = bufferFlags;

        opContext->input = new NnByte *[inputsPtr[opIndex].size()];
        opContext->inputSize = inputSizes[opIndex];
        opContext->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);
        std::memcpy(opContext->input, inputsPtr[opIndex].data(), inputsPtr[opIndex].size() * sizeof(NnByte *));

        opContext->output = new NnByte *[outputsPtr[opIndex].size()];
        opContext->outputSize = outputSizes[opIndex];
        opContext->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);
        std::memcpy(opContext->output, outputsPtr[opIndex].data(), outputsPtr[opIndex].size() * sizeof(NnByte *));

#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (opContext->weightSize.nBytes > 0)
            opContext->weight = allocAlignedBuffer(opContext->weightSize.nBytes);
        else
            opContext->weight = nullptr;
#endif

        if (opInit != nullptr)
            opInit(opContext);
        opForward[opIndex] = opForwardLocal[opIndex];
    }
    return new NnCpuDeviceSegment(opForward, opContexts, segmentConfig->nOps);
}

NnCpuDeviceSegment::~NnCpuDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < nOps; opIndex++) {
        NnCpuOpContext *context = &opContexts[opIndex];
        delete[] context->input;
        delete[] context->output;
#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (context->weightSize.nBytes > 0)
            releaseAlignedBuffer(context->weight);
#endif
    }
    delete[] opForward;
    delete[] opContexts;
}

std::vector<NnByte *> NnCpuDevice::resolvePointer(NnSize3D *pntrSize, NnPointerConfig *pointerConfig) {
    NnByte *source;
    NnSize3D *sourceSize;

    switch (pointerConfig->source) {
    case SRC_BUFFER:
        source = buffers[pointerConfig->pointerIndex];
        sourceSize = &nodeConfig->buffers[pointerConfig->pointerIndex].size;
        break;
    case SRC_PIPE:
        source = netExecution->pipes[pointerConfig->pointerIndex];
        sourceSize = &netConfig->pipes[pointerConfig->pointerIndex].size;
        break;
    default:
        throw std::invalid_argument("Unsupported pointer type");
    }

    switch (pointerConfig->type) {
    case PNTR_RAW: {
        *pntrSize = size1D(sourceSize->floatType, sourceSize->length);
        return std::vector<NnByte *>{source};
    }
    case PNTR_BATCH:
    case PNTR_BATCHED_SLICE: {
        ASSERT_EQ(sourceSize->y, netConfig->nBatches);
        std::vector<NnByte *> pntr(sourceSize->z * sourceSize->y);

        NnSize batchBytes = getBytes(sourceSize->floatType, sourceSize->x);
        for (NnUint z = 0u; z < sourceSize->z; z++) {
            for (NnUint y = 0u; y < sourceSize->y; y++)
                pntr[z * sourceSize->y + y] = &source[(z * sourceSize->y + y) * batchBytes];
        }
        *pntrSize = *sourceSize;

        if (pointerConfig->type == PNTR_BATCHED_SLICE) {
            // ====================================================
            // [重写] 智能非均匀切分逻辑
            // ====================================================
            NnUint myOffset = 0;
            NnUint myLength = 0;
            bool splitFound = false;

            // 1. 尝试查阅 Partition Plan 来获取精确的非均匀 Offset/Length
            if (partitionPlan != nullptr && netConfig->nNodes == partitionPlan->nNodes) {
                NnUint totalDim = sourceSize->x; // 管道的总维度
                NnUint nodeIdx = nodeConfig->nodeIndex;
                
                // Lambda: 检查给定的 split 是否匹配当前维度
                auto tryApplySplit = [&](const NnDimSplit& split) -> bool {
                    NnUint splitTotal = getSplitTotal(&split, partitionPlan->nNodes);
                    if (splitTotal > 0 && totalDim % splitTotal == 0) {
                        // 命中！计算倍率 (例如 HeadDim) 并应用
                        NnUint multiplier = totalDim / splitTotal;
                        myOffset = split.starts[nodeIdx] * multiplier;
                        myLength = split.lengths[nodeIdx] * multiplier;
                        return true;
                    }
                    return false;
                };

                // 按优先级尝试匹配 (Vocab > FFN > Heads)
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->vocabSplit);
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->ffnSplit);
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->headSplit);
                if (!splitFound) splitFound = tryApplySplit(partitionPlan->kvHeadSplit);
            }

            // 2. 如果没有 Plan 或没找到匹配，回退到 Legacy 均匀切分
            if (!splitFound) {
                // 移除严格断言 assert(sourceSize->x % netConfig->nNodes == 0);
                myLength = sourceSize->x / netConfig->nNodes;
                myOffset = myLength * nodeConfig->nodeIndex;
            }

            // 3. 应用偏移量 (带越界保护)
            NnSize offsetBytes = getBytes(sourceSize->floatType, myOffset);
            NnSize totalBytes = getBytes(sourceSize->floatType, sourceSize->x);
            
            if (offsetBytes >= totalBytes) {
                offsetBytes = 0;
                myLength = 0;
            }

            for (NnUint z = 0u; z < sourceSize->z; z++) {
                for (NnUint y = 0u; y < sourceSize->y; y++)
                    pntr[z * sourceSize->y + y] = &pntr[z * sourceSize->y + y][offsetBytes];
            }
            
            // 更新 size 为实际计算出的 length
            *pntrSize = size3D(sourceSize->floatType, sourceSize->z, sourceSize->y, myLength);
        }
        return pntr;
    }
    default:
        throw std::invalid_argument("Unsupported pointer config");
    }
}

void NnCpuDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex >= 0u);
    assert(opIndex < nOps);
    NnCpuOpContext *context = &opContexts[opIndex];
    if (offset + nBytes > context->weightSize.nBytes) {
        std::cerr << "🚨 CRITICAL ERROR in loadWeight:" << std::endl;
        std::cerr << "   Op Name: " << (context->name ? context->name : "Unknown") << std::endl;
        std::cerr << "   Op Index: " << opIndex << std::endl;
        std::cerr << "   Offset: " << offset << std::endl;
        std::cerr << "   Write Bytes: " << nBytes << std::endl;
        std::cerr << "   Required (Offset + Bytes): " << (offset + nBytes) << std::endl;
        std::cerr << "   Allocated Size: " << context->weightSize.nBytes << std::endl;
        std::cerr << "   Diff: " << (long long)(offset + nBytes) - (long long)context->weightSize.nBytes << std::endl;
    }
    assert(offset + nBytes <= context->weightSize.nBytes);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(offset == 0u);
    context->weight = weight;
#else
    std::memcpy(&context->weight[offset], weight, nBytes);
#endif
}

void NnCpuDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    NnCpuOpContext *context = &opContexts[opIndex];
    // printf("forward: %d %s (%d/%d)\n", opIndex, context->name, threadIndex + 1, nThreads); fflush(stdout);
    opForward[opIndex](nThreads, threadIndex, batchSize, context);
}
