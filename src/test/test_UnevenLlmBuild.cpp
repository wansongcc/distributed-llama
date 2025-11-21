#include "app.hpp" // (需要) 用于 AppCliArgs
#include "llm.hpp" // (需要) 用于 LlmHeader, LlmNet, buildLlmNetUneven, releaseLlmNet, loadLlmHeader
#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp" // (需要) 用于 NnNodeConfig
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric> // for std::accumulate
#include <cmath>   // for std::round
#include <cassert>
#include <stdexcept>
#include <cstring>   // for std::strcmp

// --- 声明 buildLlmNetUneven, 否则链接器找不到它 ---
// (这些函数在 llm.cpp 和 nn-core.cpp 中实现)
LlmNet buildLlmNetUneven(LlmHeader *h, NnUint nNodes, NnUint nBatches, const std::vector<float>& ratios);
NnUnevenPartitionPlan createPartitionPlan(
    NnUint nNodes, const std::vector<float>& ratios,
    NnUint globalNHeads, NnUint globalNKvHeads,
    NnUint globalVocabSize, NnUint globalFfnDim
);
void releasePartitionPlan(NnUnevenPartitionPlan* plan);
// (我们不需要声明 Slicers, 它们是内部实现细节)

/**
 * @brief (辅助函数) 从 app.cpp 复制而来, 用于解析 --ratios 字符串
 */
static std::vector<float> parseRatios(const char *ratiosStr, NnUint nNodes) {
    if (ratiosStr == nullptr) {
        throw std::invalid_argument("Ratios 字符串不能为空");
    }
    std::vector<float> ratios;
    std::string s(ratiosStr);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            ratios.push_back(std::stof(item));
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("无效的比例值: ") + item);
        }
    }
    if (ratios.size() != nNodes) {
        throw std::invalid_argument(
            std::string("Ratios 数量 (") + std::to_string(ratios.size()) + 
            std::string(") 必须等于节点总数 (nNodes = ") + std::to_string(nNodes) + ")"
        );
    }
    return ratios;
}

/**
 * @brief (辅助函数) 从 test_UnevenSlice.cpp 复制而来, 用于计算预期长度
 */
static NnUint calculateDimSplit(NnUint totalDim, const std::vector<float>& ratios, NnUint nodeIndex) {
    const NnUint nNodes = ratios.size();
    const float totalRatio = std::accumulate(ratios.begin(), ratios.end(), 0.0f);
    if (totalRatio <= 0.0f) throw std::invalid_argument("Total ratio must be > 0");
    
    float startRatio = 0.0f;
    for (NnUint i = 0; i < nodeIndex; ++i) startRatio += ratios[i];
    NnUint targetStart = static_cast<NnUint>(std::round(totalDim * (startRatio / totalRatio)));

    NnUint targetEnd;
    if (nodeIndex == nNodes - 1) {
        targetEnd = totalDim;
    } else {
        float endRatio = startRatio + ratios[nodeIndex];
        targetEnd = static_cast<NnUint>(std::round(totalDim * (endRatio / totalRatio)));
    }
    return targetEnd - targetStart;
}

/**
 * @brief (辅助函数) 从 test_UnevenBuilder.cpp 复制而来
 */
static const NnBufferConfig* findBufferConfig(const NnNodeConfig* nodeConfig, const char* name) {
    if (nodeConfig == nullptr) return nullptr;
    for (NnUint i = 0; i < nodeConfig->nBuffers; ++i) {
        if (std::strcmp(nodeConfig->buffers[i].name, name) == 0) {
            return &nodeConfig->buffers[i];
        }
    }
    std::cerr << "错误: 无法在节点 " << nodeConfig->nodeIndex << " 中找到缓冲区 '" << name << "'" << std::endl;
    return nullptr;
}

// (辅助函数) 获取 FFN dim, 从 llm.cpp 复制
static NnUint getFfnDim(LlmHeader* h) {
    if (h->archType == QWEN3_MOE) {
        return h->moeHiddenDim;
    }
    return h->hiddenDim;
}


// --- 主测试函数 ---
int main(int argc, char **argv) {
    std::cout << "--- 开始 [buildLlmNetUneven] 真实模型集成测试 ---" << std::endl;
    
    AppCliArgs args;
    LlmNet net = {};
    
    try {
        // 1. --- 解析命令行参数 ---
        // (使用您在 app.cpp 中已有的解析器)
        args = AppCliArgs::parse(argc, argv, false);
        std::cout <<args.ratiosStr << std::endl;
        if (args.help) {
            std::cout << "用法: ./uneven-build-live-test --model <路径> --ratios <比例>" << std::endl;
            return 0;
        }
        if (args.modelPath == nullptr) {
            throw std::runtime_error("必须提供 --model");
        }
        if (args.ratiosStr == nullptr) {
            throw std::runtime_error("必须提供 --ratios (例如 \"1,1\" 或 \"1.0,3.0\")");
        }
        NnUint nNodes = 3; // (nWorkers 默认为 0)

        // 2. --- 加载真实模型头文件 ---
        std::cout << "加载模型头文件: " << args.modelPath << std::endl;
        LlmHeader header = loadLlmHeader(args.modelPath, args.maxSeqLen, args.syncType);
        
        // (我们需要手动设置 qDim/kvDim, 因为 loadLlmHeader 可能不会设置它们)
        header.qDim = header.nHeads * header.headDim;
        header.kvDim = header.nKvHeads * header.headDim;
        NnUint ffDim = getFfnDim(&header);

        // 3. --- 解析比例 ---
        std::vector<float> ratios = parseRatios(args.ratiosStr, nNodes);
        
        // 4. --- 执行: 调用 buildLlmNetUneven ---
        std::cout << "使用 " << nNodes << " 个节点和比例 " << args.ratiosStr << " 调用 buildLlmNetUneven..." << std::endl;
        net = buildLlmNetUneven(&header, nNodes, args.nBatches, ratios);
        std::cout << "  [通过] buildLlmNetUneven 成功返回" << std::endl;

        // 5. --- 验证 (Assertions) ---
        assert(net.netConfig.nNodes == nNodes);
        assert(net.nodeConfigs != nullptr);

        for (NnUint i = 0; i < nNodes; ++i) {
            const NnNodeConfig* node = &net.nodeConfigs[i];
            assert(node != nullptr && node->nodeIndex == i);

            std::cout << "\n验证节点 " << i << " (比例: " << ratios[i] << ")..." << std::endl;

            // --- (已修复) 计算此节点的预期 (Expected) 维度 ---
            // **使用与 Builder 相同的逻辑**
            
            // 1. 切分 "头" 的数量
            NnUint expected_Heads = calculateDimSplit(header.nHeads, ratios, i);
            NnUint expected_KvHeads = calculateDimSplit(header.nKvHeads, ratios, i);
            
            // 2. 将 "头" 转换为 "维度"
            NnUint expected_qLen = expected_Heads * header.headDim;
            NnUint expected_kLen = expected_KvHeads * header.headDim;

            // 3. 切分 FFN 和 Vocab (它们不依赖 "头")
            NnUint expected_ffnLen = calculateDimSplit(ffDim, ratios, i);
            NnUint expected_vocabLen = calculateDimSplit(header.vocabSize, ratios, i);

            std::cout << "  预期: qLen=" << expected_qLen << " kLen=" << expected_kLen 
                      << " ffnLen=" << expected_ffnLen << " vocabLen=" << expected_vocabLen << std::endl;

            // --- 查找此节点的实际 (Actual) 缓冲区 ---
            const NnBufferConfig* actual_qBuf = findBufferConfig(node, "q");
            const NnBufferConfig* actual_kBuf = findBufferConfig(node, "k_temp");
            const NnBufferConfig* actual_dBuf = findBufferConfig(node, "d"); // "d" 缓冲区用于 FFN
            const NnBufferConfig* actual_lgBuf = findBufferConfig(node, "lg"); // "lg" 缓冲区用于 logits

            // --- 断言大小匹配 ---
            assert(actual_qBuf != nullptr && actual_qBuf->size.x == expected_qLen);
            assert(actual_kBuf != nullptr && actual_kBuf->size.x == expected_kLen);
            assert(actual_dBuf != nullptr && actual_dBuf->size.x == expected_ffnLen);
            assert(actual_lgBuf != nullptr && actual_lgBuf->size.x == expected_vocabLen);

            std::cout << "  [通过] 节点 " << i << " 缓冲区大小验证" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        if (net.nodeConfigs) {
            releaseLlmNet(&net); // 即使失败也要尝试清理
        }
        return 1;
    }
    
    // 6. --- 清理 ---
    std::cout << "\n清理 LlmNet..." << std::endl;
    releaseLlmNet(&net);
    std::cout << "  [通过] LlmNet 清理" << std::endl;
    
    std::cout << "\n--- 所有 [buildLlmNetUneven] 真实模型测试通过 ---" << std::endl;
    return 0;
}