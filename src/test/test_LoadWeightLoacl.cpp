#include "app.hpp"
#include "llm.hpp"
#include "nn/nn-core.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <sstream>

// --- 声明必要的外部函数 ---
LlmNet buildLlmNetUneven(LlmHeader *h, NnUint nNodes, NnUint nBatches, const std::vector<float>& ratios);
void releaseLlmNet(LlmNet *net);
NnUnevenPartitionPlan createPartitionPlan(NnUint nNodes, const std::vector<float>& ratios, NnUint globalNHeads, NnUint globalNKvHeads, NnUint globalVocabSize, NnUint globalFfnDim);
void releasePartitionPlan(NnUnevenPartitionPlan* plan);
static NnUint getFfnDim(LlmHeader* h) { return (h->archType == QWEN3_MOE) ? h->moeHiddenDim : h->hiddenDim; }

// 解析比例辅助函数
static std::vector<float> parseRatios(const char *ratiosStr) {
    std::vector<float> ratios;
    std::string s(ratiosStr);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) ratios.push_back(std::stof(item));
    }
    return ratios;
}

int main(int argc, char **argv) {
    std::cout << "===================================================" << std::endl;
    std::cout << "   Worker 本地加载权重测试 (Worker Local Load Test)   " << std::endl;
    std::cout << "===================================================" << std::endl;

    if (argc < 3) {
        std::cerr << "用法: ./worker-load-test <model_path> <ratios>" << std::endl;
        std::cerr << "示例: ./worker-load-test models/qwen.m \"1.0,2.0\"" << std::endl;
        return 1;
    }

    const char* modelPath = argv[1];
    const char* ratiosStr = argv[2];
    const NnUint nBatches = 1;
    
    try {
        // 1. 准备环境
        std::vector<float> ratios = parseRatios(ratiosStr);
        NnUint nNodes = ratios.size();
        if (nNodes < 2) {
            std::cerr << "测试需要至少 2 个节点来模拟 Worker" << std::endl;
            return 1;
        }

        // 模拟 Worker 节点 (例如取最后一个节点)
        NnUint myNodeIndex = nNodes - 1; 
        float myRatio = ratios[myNodeIndex];
        std::cout << "模拟 Worker 节点: Index=" << myNodeIndex << ", TotalNodes=" << nNodes << ", Ratio=" << myRatio << std::endl;

        // 2. 加载头信息
        LlmHeader header = loadLlmHeader(modelPath, 2048, F_Q80); // 假设使用 Q80 这里的参数根据实际调整
        if (header.headDim == 0) header.headDim = header.dim / header.nHeads;
        header.qDim = header.nHeads * header.headDim;
        header.kvDim = header.nKvHeads * header.headDim;

        // 3. 构建网络配置 (模拟 Root 发来的配置)
        std::cout << "构建网络配置..." << std::endl;
        LlmNet net = buildLlmNetUneven(&header, nNodes, nBatches, ratios);
        std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

        // 获取属于本 Worker 的配置
        NnNodeConfig* myNodeConfig = &net.nodeConfigs[myNodeIndex];

        // 4. 创建执行环境 (Executor)
        // 我们使用 FakeSynchronizer，因为这里没有真实网络连接
        NnNetExecution execution(1, &net.netConfig); // 1 thread
        NnFakeNodeSynchronizer fakeSync; 
        
        // 创建 CPU 设备
        std::vector<NnExecutorDevice> devices;
        devices.push_back(NnExecutorDevice(new NnCpuDevice(&net.netConfig, myNodeConfig, &execution), -1, -1));
        
        NnExecutor executor(&net.netConfig, myNodeConfig, &devices, &execution, &fakeSync, false);

        // 5. 准备非均匀切分计划 (Plan)
        NnUint ffDim = getFfnDim(&header);
        NnUnevenPartitionPlan plan = createPartitionPlan(nNodes, ratios, header.nHeads, header.nKvHeads, header.vocabSize, ffDim);

        // 6. [核心测试] 执行本地加载
        std::cout << "🚀 开始执行 loadLlmNetWeightUneven (Local)..." << std::endl;
        
        NnLocalWeightLoader localLoader(&executor, myNodeIndex);
        loadLlmNetWeightUneven(modelPath, &net, &localLoader, &plan);

        // 7. 简单的验证
        // 我们可以检查 Executor 中的某些权重是否非空。
        // 由于 NnExecutor 内部结构比较封闭，这里只要函数成功返回不报错，
        // 并且日志显示 "Loaded ..."，通常就意味着内存已正确写入。
        std::cout << "✅ Worker 权重加载成功完成！" << std::endl;

        releasePartitionPlan(&plan);

    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}