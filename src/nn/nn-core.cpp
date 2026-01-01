#ifdef _WIN32
    #define _USE_MATH_DEFINES
#endif
#include "nn-core.hpp"
#include "nn-quants.hpp"
#include <cassert>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>     
#include <numeric>    

// utility functions

NnSize getBytes(NnFloatType floatType, NnSize n) {
    if (floatType == F_32)
        return n * sizeof(float);
    if (floatType == F_16)
        return n * (sizeof(float) / 2);
    if (floatType == F_Q40) {
        assert(n % Q40_BLOCK_SIZE == 0);
        return (n / Q40_BLOCK_SIZE) * sizeof(NnBlockQ40);
    }
    if (floatType == F_Q80) {
        assert(n % Q80_BLOCK_SIZE == 0);
        return (n / Q80_BLOCK_SIZE) * sizeof(NnBlockQ80);
    }
    throw std::invalid_argument("Unsupported float type: " + std::to_string(floatType));
}

NnSize getBlockSize(NnFloatType floatType) {
    if (floatType == F_32)
        return 1;
    if (floatType == F_16)
        return 1;
    if (floatType == F_Q40)
        return Q40_BLOCK_SIZE;
    if (floatType == F_Q80)
        return Q80_BLOCK_SIZE;
    throw std::invalid_argument("Unsupported float type");
}

NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output) {
    // If weight=F_UNK, then returned enum should be <input>_<input>_<output>

    if (input == F_32 && output == F_32) {
        if (weight == F_UNK || weight == F_32)
            return F32_F32_F32;
        if (weight == F_Q40)
            return F32_Q40_F32;
    }
    if (input == F_32 && output == F_Q80) {
        if (weight == F_UNK || weight == F_32)
            return F32_F32_Q80;
        if (weight == F_Q40)
            return F32_Q40_Q80;
    }
    if (input == F_Q80 && output == F_32) {
        if (weight == F_UNK || weight == F_Q80)
            return Q80_Q80_F32;
        if (weight == F_32)
            return Q80_F32_F32;
        if (weight == F_Q40)
            return Q80_Q40_F32;
    }
    if (input == F_Q80 && output == F_Q80) {
        if (weight == F_UNK || weight == F_Q80)
            return Q80_Q80_Q80;
    }
    throw std::invalid_argument("Unsupported op quant: " + 
        std::string(floatTypeToString(input)) + "/" +
        std::string(floatTypeToString(weight)) + "/" +
        std::string(floatTypeToString(output)));
}

const char *opCodeToString(NnOpCode code) {
    if (code == OP_MERGE_ADD) return "MERGE_ADD";
    if (code == OP_MERGE_SUM) return "MERGE_SUM";
    if (code == OP_EMBEDDING) return "EMBEDDING";
    if (code == OP_INV_RMS) return "INV_RMS";
    if (code == OP_RMS_NORM) return "RMS_NORM";
    if (code == OP_MATMUL) return "MATMUL";
    if (code == OP_ROPE) return "ROPE";
    if (code == OP_MULTIHEAD_ATT) return "MULTIHEAD_ATT";
    if (code == OP_GELU) return "GELU";
    if (code == OP_SILU) return "SILU";
    if (code == OP_MUL) return "MUL";
    if (code == OP_SCALE) return "SCALE";
    if (code == OP_CAST) return "CAST";
    if (code == OP_REPEAT_Z) return "REPEAT_Z";
    if (code == OP_SHIFT) return "SHIFT";
    if (code == OP_SOFTMAX) return "SOFTMAX";
    if (code == OP_MOE_GATE) return "MOE_GATE";
    throw std::invalid_argument("Unknown op code: " + std::to_string(code));
}

const char *opQuantTypeToString(NnOpQuantType type) {
    if (type == F32_F32_F32) return "F32_F32_F32";
    if (type == F32_Q40_F32) return "F32_Q40_F32";
    if (type == F32_Q40_Q80) return "F32_Q40_Q80";
    if (type == F32_F32_Q80) return "F32_F32_Q80";
    if (type == Q80_Q80_Q80) return "Q80_Q80_Q80";
    if (type == Q80_Q80_F32) return "Q80_Q80_F32";
    if (type == Q80_Q40_F32) return "Q80_Q40_F32";
    if (type == Q80_F32_F32) return "Q80_F32_F32";
    throw std::invalid_argument("Unknown op quant type");
}

NnSize3D size0() {
    return { F_UNK, 0, 0, 0, 0, 0 };
}

NnSize3D size1D(NnFloatType floatType, NnUint x) {
    return size3D(floatType, 1, 1, x);
}

NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x) {
    return size3D(floatType, 1, y, x);
}

NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x) {
    NnSize len = z * y * x;
    NnSize lenXY = y * x;
    return { floatType, z, y, x, len, getBytes(floatType, len), getBytes(floatType, lenXY) };
}

NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_BATCH };
}

NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_BATCHED_SLICE };
}

NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_RAW };
}

bool hasPointerContinuousMemory(NnPointerConfig *config) {
    if (config->type == PNTR_RAW)
        return true;
    if (config->type == PNTR_BATCH)
        return true;
    return false;
}

void releaseNetConfig(NnNetConfig *netConfig) {
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        delete[] netConfig->pipes[pipeIndex].name;
    }
    if (netConfig->nPreSyncs > 0)
        delete[] netConfig->preSyncs;
    delete[] netConfig->pipes;
}

void releaseNodeConfig(NnNodeConfig *nodeConfig) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        if (segment->nOps > 0) {
            for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
                NnOpConfig *op = &segment->ops[opIndex];
                delete[] op->name;
                delete[] op->config;
            }
            delete[] segment->ops;
        }
        if (segment->nSyncs > 0)
            delete[] segment->syncs;
    }
    if (nodeConfig->nBuffers > 0) {
        for (NnUint bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
            delete[] nodeConfig->buffers[bufferIndex].name;
        delete[] nodeConfig->buffers;
    }
    delete[] nodeConfig->segments;
}

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    unsigned long total = 0;
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++)
        total += netConfig->pipes[pipeIndex].size.nBytes;
    for (NnUint bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
        total += nodeConfig->buffers[bufferIndex].size.nBytes;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
            total += segment->ops[opIndex].weightSize.nBytes;
            total += segment->ops[opIndex].configSize;
        }
    }
    printf("📀 RequiredMemory: %lu MB\n", total / (1024 * 1024));
}

Timer::Timer() {
    reset();
}

void Timer::reset() {
    startTime = std::chrono::high_resolution_clock::now();
}

NnUint Timer::elapsedMiliseconds() {
    auto endTime = std::chrono::high_resolution_clock::now();
    return (NnUint)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

NnUint Timer::elapsedMicroseconds() {
    auto endTime = std::chrono::high_resolution_clock::now();
    return (NnUint)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes) {
    NnKvCacheSlice s;
    assert(kvDim % nNodes == 0);
    s.kvDim0 = kvDim / nNodes;
    s.keySize = size2D(F_32, seqLen, s.kvDim0);
    s.valueSize = size2D(F_32, seqLen, s.kvDim0);
    return s;
}

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    assert(d % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.d0 = d / nNodes;
    s.n = n;
    s.size = size2D(type, s.n, d);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}

NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnColMatmulSlice s;
    assert(n % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.n = n;
    s.n0 = n / nNodes;
    s.d = d;
    s.size = size2D(type, n, d);
    s.sliceSize = size2D(type, s.n0, d);
    return s;
}

NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex) {
    NnRopeSlice s;
    assert(qDim >= kvDim);
    assert(qDim % nNodes == 0);
    assert(kvDim % nNodes == 0);

    s.kvDim = kvDim;
    s.nKvHeads = nKvHeads;
    s.seqLen = seqLen;
    s.headDim = headDim;
    s.ropeTheta = ropeTheta;

    s.qDim0 = qDim / nNodes;
    s.kvDim0 = kvDim / nNodes;
    assert(s.qDim0 % 2 == 0);
    assert(s.kvDim0 % 2 == 0);

    if (type == ROPE_LLAMA || type == ROPE_LLAMA3_1) {
        s.kvDimStart = s.kvDim0 * nodeIndex;
        s.qDimStart = s.qDim0 * nodeIndex;
        s.qDimEnd = s.qDimStart + s.qDim0;
        s.qShift = s.qDimStart - s.kvDimStart;
        s.sliceDim = s.qDimEnd - s.kvDimStart;
        assert(s.sliceDim % 2 == 0);
        s.cacheSize = size2D(F_32, seqLen, s.sliceDim);
    } else if (type == ROPE_FALCON) {
        s.cacheSize = size2D(F_32, seqLen, headDim);
    } else {
        throw std::invalid_argument("Unsupported rope type");
    }
    return s;
}


NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches) {
    NnMultiHeadAttSlice s;
    assert(nHeads % nNodes == 0);
    s.nHeads = nHeads;
    s.nHeads0 = nHeads / nNodes;
    s.attSize = size2D(F_32, nBatches, s.nHeads0 * seqLen);
    return s;
}

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize offset = slice->d0 * nodeIndex * n * batchBytes;
    NnSize copiedBytes = 0;
    for (NnUint d = 0; d < slice->d0; d++) {
        for (NnUint j = 0; j < n; j++) {
            NnSize o = (d * n + j) * batchBytes;
            std::memcpy(weight0 + o, weight + offset + o, batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes;
}

NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n0 % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize rowBytes = n * batchBytes;
    NnSize row0Bytes = (slice->n0 / blockSize) * batchBytes;
    NnSize rowOffsetBytes = nodeIndex * row0Bytes;
    NnSize copiedBytes = 0;
    for (NnUint d = 0; d < slice->d; d++) {
        std::memcpy(&weight0[row0Bytes * d], &weight[rowBytes * d + rowOffsetBytes], row0Bytes);
        copiedBytes += row0Bytes;
    }
    return copiedBytes;
}

//Uneven slicers
NnDimSplit createDimSplit(NnUint totalDim, const std::vector<float>& ratios) {
    NnUint nNodes = ratios.size();
    if (nNodes == 0) {
        throw std::invalid_argument("Ratios vector cannot be empty.");
    }

    NnUint* starts = new NnUint[nNodes];
    NnUint* lengths = new NnUint[nNodes];

    // 1. calculate total ratio sum
    const float totalRatio = std::accumulate(ratios.begin(), ratios.end(), 0.0f);
    if (totalRatio <= 0.0f) {
        delete[] starts;
        delete[] lengths;
        throw std::invalid_argument("Total ratio must be greater than 0");
    }

    float cumulativeRatio = 0.0f;
    NnUint currentOffset = 0;

    // 2. iterate over each node to calculate its 'start' and 'length'
    for (NnUint i = 0; i < nNodes; ++i) {
        starts[i] = currentOffset;

        if (i == nNodes - 1) {
            // last node: assign all remaining dimensions to ensure total sum matches exactly
            lengths[i] = totalDim - currentOffset;
        } else {
            // calculate this node's *target* end point
            cumulativeRatio += ratios[i];
            NnUint targetEnd = static_cast<NnUint>(
                std::round(totalDim * (cumulativeRatio / totalRatio))
            );
            
            lengths[i] = (targetEnd > currentOffset) ? (targetEnd - currentOffset) : 0;
        }
        currentOffset += lengths[i];
    }

    if (currentOffset != totalDim && nNodes > 0) {
         delete[] starts;
         delete[] lengths;
         throw std::runtime_error("createDimSplit logic error: sum does not match totalDim.");
    }

    return NnDimSplit{starts, lengths};
}

// 参数 offset: 全局节点 ID 的起始偏移 (例如 Stage 1 从 Node 2 开始)
static void fillDimSplitForStage(NnDimSplit& split, NnUint offset, NnUint totalDim, const std::vector<float>& ratios, NnUint alignSize) {
    NnUint nLocalNodes = ratios.size();
    
    float ratioSum = 0;
    for(float r : ratios) ratioSum += r;
    if (ratioSum < 1e-6) throw std::invalid_argument("Ratio sum is too small");

    NnUint currentStart = 0; 
    NnUint remainingDim = totalDim;

    for(NnUint i = 0; i < nLocalNodes; i++) {
        NnUint globalNodeIdx = offset + i;
        
        split.starts[globalNodeIdx] = currentStart;
        
        NnUint len;
        if(i == nLocalNodes - 1) {
            len = remainingDim; 
        } else {
            double ideal = (double)totalDim * (ratios[i] / ratioSum);
            len = (NnUint)round(ideal);
            
            // [修改] 使用传入的 alignSize 进行对齐
            if (alignSize > 1) {
                NnUint remainder = len % alignSize;
                if (remainder != 0) {
                    if (remainder >= alignSize / 2) {
                        len += (alignSize - remainder);
                    } else {
                        // 防止变成 0 (除非 alignSize 本身很大而 ideal 很小，
                        // 但对于 Dim 来说通常够大。对于 Head 来说 alignSize=1 不会进这里)
                        if (len > remainder) len -= remainder;
                    }
                }
                // 极小值保护：如果是维度切分，至少保留一个块
                if (len == 0 && totalDim >= nLocalNodes * alignSize) len = alignSize;
            }
            
            if (len > remainingDim) len = remainingDim;
        }
        
        split.lengths[globalNodeIdx] = len;
        
        currentStart += len;
        remainingDim -= len;
    }
}

NnUnevenPartitionPlan createPartitionPlan(
    const std::vector<NnStageDef>& stageDefs,
    NnUint globalNHeads,
    NnUint globalNKvHeads,
    NnUint globalVocabSize,
    NnUint globalFfnDim,
    NnUint globalDim
) {
    NnUnevenPartitionPlan plan;
    
    // 1. 基础校验与统计
    if (stageDefs.empty()) throw std::invalid_argument("No stages defined");
    
    plan.nStages = stageDefs.size();
    plan.nNodes = 0;
    for (const auto& stage : stageDefs) {
        if (stage.tpRatios.empty()) throw std::invalid_argument("Stage must have nodes");
        plan.nNodes += stage.tpRatios.size();
    }

    // 2. 初始化全局数组
    plan.stages = new NnStageConfig[plan.nStages];
    
    // Helper to allocate split arrays
    auto allocSplit = [&](NnDimSplit& s) {
        s.starts = new NnUint[plan.nNodes];
        s.lengths = new NnUint[plan.nNodes];
        std::memset(s.starts, 0, plan.nNodes * sizeof(NnUint));
        std::memset(s.lengths, 0, plan.nNodes * sizeof(NnUint));
    };
    
    allocSplit(plan.headSplit);
    allocSplit(plan.kvHeadSplit);
    allocSplit(plan.vocabSplit);
    allocSplit(plan.ffnSplit);
    allocSplit(plan.dimSplit);

    // GQA Check
    if (globalNHeads % globalNKvHeads != 0) {
        throw std::runtime_error("nHeads must be divisible by nKvHeads");
    }
    NnUint gqaGroupSize = globalNHeads / globalNKvHeads;

    try {
        NnUint currentNodeOffset = 0;
        NnUint currentLayerOffset = 0;

        // 3. 逐个 Stage 生成配置
        for (NnUint s = 0; s < plan.nStages; s++) {
            const NnStageDef& def = stageDefs[s];
            NnStageConfig& config = plan.stages[s];

            // 3.1 填充 Stage 拓扑信息
            config.stageIndex = s;
            config.startLayer = currentLayerOffset;
            config.nLayers = def.nLayers;
            config.endLayer = config.startLayer + config.nLayers;
            
            config.nNodes = def.tpRatios.size();
            config.rootNodeIndex = currentNodeOffset; // 默认 Stage 的第一个节点是 Root
            config.nodeIndices = new NnUint[config.nNodes];
            for (NnUint i = 0; i < config.nNodes; i++) {
                config.nodeIndices[i] = currentNodeOffset + i;
            }

            // 3.2 填充该 Stage 内的 TP Split
            // 注意：每个 Stage 都是一个独立的 TP 组，所以维度必须在该 Stage 内完整分配
            
            // KV Heads
            fillDimSplitForStage(plan.kvHeadSplit, currentNodeOffset, globalNKvHeads, def.tpRatios, 1);
            
            // Q Heads (GQA 对齐)
            // 基于刚刚生成的 KV Split 计算 Q Split
            for (NnUint i = 0; i < config.nNodes; i++) {
                NnUint globalIdx = currentNodeOffset + i;
                plan.headSplit.starts[globalIdx] = plan.kvHeadSplit.starts[globalIdx] * gqaGroupSize;
                plan.headSplit.lengths[globalIdx] = plan.kvHeadSplit.lengths[globalIdx] * gqaGroupSize;
            }

            // FFN & Dim (Hidden Size)
            fillDimSplitForStage(plan.ffnSplit, currentNodeOffset, globalFfnDim, def.tpRatios, 32);
            fillDimSplitForStage(plan.dimSplit, currentNodeOffset, globalDim, def.tpRatios, 32);

            // Vocab (Logits)
            // 虽然只有 Last Stage 真正计算 Logits，但为了逻辑统一，
            // 我们为所有 Stage 都计算 Vocab Split (Loader 会根据层号自动跳过非 Logits 层)
            fillDimSplitForStage(plan.vocabSplit, currentNodeOffset, globalVocabSize, def.tpRatios, 32);

            // 推进偏移量
            currentNodeOffset += config.nNodes;
            currentLayerOffset += config.nLayers;
        }

    } catch (...) {
        // NnUnevenPartitionPlan 析构函数会处理内存释放
        throw;
    }

    return plan;
}

    
NnKvCacheSliceUneven sliceKvCacheUneven(NnUint seqLen, NnUint headDim,
                                        const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    NnKvCacheSliceUneven s;

    // 1. 从“总蓝图”中查询本节点的 KV Head 分配
    const NnUint kvHeadStart = plan->kvHeadSplit.starts[nodeIndex];
    const NnUint kvHeadLen = plan->kvHeadSplit.lengths[nodeIndex];

    // 2. 将 Head 分配转换为维度 (Start/Length)
    s.kvStart = kvHeadStart * headDim;
    s.kvLen = kvHeadLen * headDim;
    
    // 3. 填充兼容性/派生字段
    s.kvDim0 = s.kvLen; // 保留以兼容旧逻辑

    // 4. 计算局部缓冲区大小 (复用 size2D)
    s.keySize = size2D(F_32, seqLen, s.kvLen);
    s.valueSize = size2D(F_32, seqLen, s.kvLen);

    return s;
}

NnMultiHeadAttSliceUneven sliceMultiHeadAttUneven(NnUint nBatches, NnUint globalNHeads, NnUint globalSeqLen,
                                                  const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    NnMultiHeadAttSliceUneven s;

    // 1. 从“总蓝图”中查询本节点的 Head 分配
    s.headStart = plan->headSplit.starts[nodeIndex];
    s.headLen = plan->headSplit.lengths[nodeIndex];

    // 2. 填充兼容性/派生字段
    s.nHeads = globalNHeads; // 全局 Head 总数
    s.nHeads0 = s.headLen;   // 局部 Head 数量

    // 3. 计算局部缓冲区大小 (复用 size2D)
    s.attSize = size2D(F_32, nBatches, s.headLen * globalSeqLen); 

    return s;
}

//uesd for q,k,v projection weight slicing
NnRowMatmulSliceUneven sliceRowMatmulAttUneven(NnFloatType type, NnUint globalInDim, NnUint headDim,
                                               const NnDimSplit* headSplit, 
                                               NnUint globalOutDim, NnUint nodeIndex) {
    NnRowMatmulSliceUneven s;
    s.type = type;

    // 1. 从 Head 蓝图中获取分配
    const NnUint headStart = headSplit->starts[nodeIndex];
    const NnUint headLen = headSplit->lengths[nodeIndex];

    // 2. 转换为维度，并填入 'inStart'/'inLen' 字段
    s.inStart = headStart * headDim;
    s.inLen = headLen * headDim;

    // 3. 填充兼容性/派生字段
    s.d0 = s.inLen;   // d0 是局部输出维度
    s.n = globalInDim; // n 是完整输入维度
    
    // 4. 计算尺寸 (复用 size2D)
    s.size = size2D(type, s.n, globalOutDim);  // 完整权重矩阵的大小
    s.sliceSize = size2D(type, s.n, s.d0);     // 本节点切片的大小

    return s;
}

//wo
NnColMatmulSliceUneven sliceColMatmulAttUneven(NnFloatType type, NnUint globalInDimQ, NnUint globalOutDim, NnUint headDim,
                                               const NnUnevenPartitionPlan* plan, 
                                               NnUint nodeIndex) {
    NnColMatmulSliceUneven s;
    s.type = type;

    // 1. 从 Head 蓝图 (headSplit) 获取分配
    const NnUint headStart = plan->headSplit.starts[nodeIndex];
    const NnUint headLen = plan->headSplit.lengths[nodeIndex];
    
    // 2. 转换为维度，并填入 'outStart'/'outLen' 字段
    s.outStart = headStart * headDim;
    s.outLen = headLen * headDim;

    // 3. 填充兼容性/派生字段
    s.n = globalInDimQ; // n 是完整输入维度
    s.n0 = s.outLen;    // n0 是局部输入维度
    s.d = globalOutDim; // d 是完整输出维度

    // 4. 计算尺寸 (复用 size2D)
    s.size = size2D(type, s.n, s.d);
    s.sliceSize = size2D(type, s.n0, s.d);

    return s;
}

//ffn
NnRowMatmulSliceUneven sliceRowMatmulFfnUneven(NnFloatType type, NnUint globalInDim, NnUint globalFfnDim,
                                               const NnUnevenPartitionPlan* plan, 
                                               NnUint nodeIndex) {
    NnRowMatmulSliceUneven s;
    s.type = type;

    // 1. 从 FFN 蓝图中获取分配 (不再乘以 headDim)
    s.inStart = plan->ffnSplit.starts[nodeIndex];
    s.inLen = plan->ffnSplit.lengths[nodeIndex];

    // 2. 填充兼容性/派生字段
    s.d0 = s.inLen;   // d0 是局部输出维度
    s.n = globalInDim; // n 是完整输入维度 (h->dim)

    // 3. 计算尺寸
    s.size = size2D(type, s.n, globalFfnDim);
    s.sliceSize = size2D(type, s.n, s.d0);

    return s;
}


NnColMatmulSliceUneven sliceColMatmulFfnUneven(NnFloatType type, NnUint globalFfnDim, NnUint globalOutDim,
                                               const NnUnevenPartitionPlan* plan, 
                                               NnUint nodeIndex) {
    NnColMatmulSliceUneven s;
    s.type = type;

    // 1. 从 FFN 蓝图中获取分配 (不再乘以 headDim)
    s.outStart = plan->ffnSplit.starts[nodeIndex];
    s.outLen = plan->ffnSplit.lengths[nodeIndex];

    // 3. 填充兼容性/派生字段
    s.n = globalFfnDim; // n 是完整输入维度
    s.n0 = s.outLen;    // n0 是局部输入维度
    s.d = globalOutDim; // d 是完整输出维度 (h->dim)

    // 4. 计算尺寸
    s.size = size2D(type, s.n, s.d);
    s.sliceSize = size2D(type, s.n0, s.d);

    return s;
}

NnRowMatmulSliceUneven sliceRowMatmulLogitsUneven(NnFloatType type, NnUint globalInDim, NnUint globalVocabSize,
    const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    NnRowMatmulSliceUneven s;
    s.type = type;
    s.inStart = plan->vocabSplit.starts[nodeIndex];
    s.inLen = plan->vocabSplit.lengths[nodeIndex];
    s.d0 = s.inLen;
    s.n = globalInDim;
    s.size = size2D(type, s.n, globalVocabSize);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}

NnRopeSliceUneven sliceRopeUneven(NnRopeType type, NnUint seqLen, 
                                  NnUint globalKvDim, NnUint globalNKvHeads, NnUint headDim, float ropeTheta,
                                  const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    NnRopeSliceUneven s;
    std::memset(&s, 0, sizeof(s));

    // --- 1. Q 侧 (来自 headSplit) ---
    const NnUint qHeadStart = plan->headSplit.starts[nodeIndex];
    s.qDimLen = plan->headSplit.lengths[nodeIndex] * headDim;
    s.qDimStart = qHeadStart * headDim;
    s.qDim0 = s.qDimLen; // 兼容字段

    // --- 2. KV 侧 (来自 kvHeadSplit) ---
    const NnUint kvHeadStart = plan->kvHeadSplit.starts[nodeIndex];
    s.kvDimLen = plan->kvHeadSplit.lengths[nodeIndex] * headDim;
    s.kvDimStart = kvHeadStart * headDim;
    s.kvDim0 = s.kvDimLen; // 兼容字段

    // --- 3. 填充其它参数 ---
    s.kvDim = globalKvDim;
    s.nKvHeads = globalNKvHeads;
    s.seqLen = seqLen;
    s.headDim = headDim;
    s.ropeTheta = ropeTheta;

    // --- 4. 计算派生字段和 Cache (复用均匀 sliceRope 逻辑) ---
    if (type == ROPE_LLAMA || type == ROPE_LLAMA3_1) {
        s.qShift = s.qDimStart - s.kvDimStart;
        NnUint qDimEnd = s.qDimStart + s.qDimLen;
        s.sliceDim = qDimEnd - s.kvDimStart; 
        assert(s.sliceDim % 2 == 0);
        s.cacheSize = size2D(F_32, seqLen, s.sliceDim);
    } else if (type == ROPE_FALCON) {
        s.sliceDim = headDim;
        s.cacheSize = size2D(F_32, seqLen, headDim);
    } else {
        throw std::invalid_argument("Unsupported rope type");
    }
    return s;
}

//Uneven sllitter weight functions
NnUint splitRowMatmulWeightUneven(NnRowMatmulSliceUneven *slice, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    
    // 校验对齐
    assert(slice->n % blockSize == 0); // n 是完整输入维度 (Width)

    // 1. 计算“一行”的字节数 (Global Stride)
    NnSize bytes_per_row = (slice->n / blockSize) * batchBytes;

    // 2. 计算源 (weight) 的起始字节偏移
    // inStart: 起始行号
    NnSize offset = slice->inStart * bytes_per_row;

    // 3. 计算本节点需要复制的总字节数
    // inLen: 本节点负责的行数
    NnSize total_copy_bytes = slice->inLen * bytes_per_row;

    // 4. 单次内存拷贝 (极快)
    // 注意：如果是 NnLocalWeightLoader 的 Zero-Copy 优化，这一步甚至都不需要
    // 但作为通用工具函数，这里应该是 memcpy
    std::memcpy(weight0, weight + offset, total_copy_bytes);

    return total_copy_bytes;
}

NnUint splitColMatmulWeightUneven(NnColMatmulSliceUneven *slice, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);

    // 1. 验证切分是块对齐的
    assert(slice->outLen % blockSize == 0); // 局部长度
    assert(slice->outStart % blockSize == 0); // 全局起始点
    assert(slice->n % blockSize == 0);      // 全局总高度

    // 2. 计算“完整”权重中一列的字节大小
    NnSize n_global_blocks = slice->n / blockSize;
    NnSize rowBytes = n_global_blocks * batchBytes; 
    
    // 3. 计算“局部”权重中一列的字节大小
    NnSize n_local_blocks = slice->outLen / blockSize;
    NnSize row0Bytes = n_local_blocks * batchBytes;

    // 4. (关键) 计算源 (weight) 中要复制的“起始行”的字节偏移量
    NnSize start_block = slice->outStart / blockSize;
    NnSize rowOffsetBytes = start_block * batchBytes;

    NnSize copiedBytes = 0;
    
    // 5. 遍历每一列 'd'
    for (NnUint d = 0; d < slice->d; d++) {
        // 目标: 局部缓冲区(weight0) 的第 'd' 列的开头
        NnByte* dest = &weight0[row0Bytes * d];
        // 源: 完整缓冲区(weight) 的第 'd' 列, 再偏移 'rowOffsetBytes' (起始行)
        NnByte* src = &weight[rowBytes * d + rowOffsetBytes];
        
        // 一次性复制这一列中属于本节点的所有行
        std::memcpy(dest, src, row0Bytes);
        copiedBytes += row0Bytes;
    }
    return copiedBytes;
}


// helper

static inline float scaleFrequencyLlama3(const float freq, const NnRopeOpConfig *config) {
    // https://github.com/meta-llama/llama-models/blob/4269717b2ea587627903bacbb75ccce1427ad914/models/llama3/reference_impl/model.py#L55
    const float waveLen = 2.0f * M_PI / freq;
    const float highFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingHighFreqFactor;
    if (waveLen < highFreqWavelen) {
        return freq;
    }
    const float lowFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingLowFreqFactor;
    if (waveLen > lowFreqWavelen) {
        return freq / config->ropeScalingFactor;
    }
    const float smooth = (config->ropeScalingOrigMaxSeqLen / waveLen - config->ropeScalingLowFreqFactor) /
        (config->ropeScalingHighFreqFactor - config->ropeScalingLowFreqFactor);
    return (1 - smooth) * freq / config->ropeScalingFactor + smooth * freq;
}

static inline void fullfillRopeLlamaCache(const NnRopeOpConfig *config, float *cache) {
    assert((config->slice.qDimEnd - config->slice.kvDimStart) % 2 == 0);

    const bool applyScaling = config->ropeScalingFactor != 1.0f;
    float theta = 1000000.0f;
    for (NnUint pos = 0; pos < config->slice.seqLen; pos++) {
        for (NnUint i = config->slice.kvDimStart; i < config->slice.qDimEnd; i += 2) {
            const NnUint h = i % config->slice.headDim;
            float freq = 1.0f / powf(theta, h / (float)config->slice.headDim);
            if (applyScaling)
                freq = scaleFrequencyLlama3(freq, config);
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * config->slice.sliceDim + (i - config->slice.kvDimStart)] = fcr;
            cache[pos * config->slice.sliceDim + (i - config->slice.kvDimStart) + 1] = fci;
        }
    }
}

static inline void fullfillRopeFalconCache(const NnRopeOpConfig *config, float *cache) {
    const float hs = (float)config->slice.headDim;

    for (NnUint pos = 0; pos < config->slice.seqLen; pos++) {
        for (NnUint j = 0; j < config->slice.headDim / 2; j++) {
            const float freq = 1.0f / powf(config->slice.ropeTheta, 2.0f * (float)(j / hs));
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * config->slice.headDim + j] = fcr;
            cache[pos * config->slice.headDim + j + config->slice.headDim / 2] = fci;
        }
    }
}

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache) {
    if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1)
        fullfillRopeLlamaCache(config, cache);
    else if (config->type == ROPE_FALCON)
        fullfillRopeFalconCache(config, cache);
    else
        throw std::invalid_argument("Unsupported rope type");
}

//release uneven partition plan
void releasePartitionPlan(NnUnevenPartitionPlan* plan) {
    if (plan == nullptr) return;

    delete[] plan->headSplit.starts;
    delete[] plan->headSplit.lengths;
    
    delete[] plan->kvHeadSplit.starts;
    delete[] plan->kvHeadSplit.lengths;

    delete[] plan->vocabSplit.starts;
    delete[] plan->vocabSplit.lengths;

    delete[] plan->ffnSplit.starts;
    delete[] plan->ffnSplit.lengths;

    // 将指针设为 null 以防止重复释放
    plan->headSplit = {nullptr, nullptr};
    plan->kvHeadSplit = {nullptr, nullptr};
    plan->vocabSplit = {nullptr, nullptr};
    plan->ffnSplit = {nullptr, nullptr};
    plan->nNodes = 0;
}