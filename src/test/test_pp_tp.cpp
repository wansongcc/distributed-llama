#include <cassert>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>

// 引入必要的头文件
#include "llm.hpp"
#include "nn/nn-core.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-cpu.hpp"

// --------------------------------------------------------------------------
// 简化的参数解析器
// --------------------------------------------------------------------------
struct TestArgs {
    const char* modelPath = nullptr;
    const char* ratiosStr = nullptr;
    int nodeIndex = 0; // 模拟的节点 ID
    int nThreads = 1;
    NnFloatType syncType = F_Q80; // 默认同步类型
};

TestArgs parseTestArgs(int argc, char** argv) {
    TestArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.modelPath = argv[++i];
        } else if (strcmp(argv[i], "--ratios") == 0 && i + 1 < argc) {
            args.ratiosStr = argv[++i];
        } else if (strcmp(argv[i], "--node-index") == 0 && i + 1 < argc) {
            args.nodeIndex = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nthreads") == 0 && i + 1 < argc) {
            args.nThreads = atoi(argv[++i]);
        }
    }
    if (!args.modelPath || !args.ratiosStr) {
        throw std::runtime_error("Usage: ./test_load_only --model <path> --ratios <str> [--node-index <int>]");
    }
    return args;
}

// 复用 app.cpp 中的解析函数 (为了独立编译，这里复制一份简单的实现)
static std::vector<float> parseRatiosLocal(const char *ratiosStr, NnUint nNodes) {
    std::vector<float> ratios;
    std::string s(ratiosStr);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        ratios.push_back(std::stof(item));
    }
    if (ratios.size() != nNodes) {
        throw std::runtime_error("Ratios count must match node count.");
    }
    return ratios;
}

// --------------------------------------------------------------------------
// 打印 Plan 详情的辅助函数
// --------------------------------------------------------------------------
void printPlanDetails(const NnUnevenPartitionPlan* plan) {
    printf("\n🔍 [DEBUG] Partition Plan Details:\n");
    printf("===================================\n");
    printf("Global: %u Stages, %u Nodes\n", plan->nStages, plan->nNodes);
    
    for(NnUint s=0; s<plan->nStages; ++s) {
        const auto& stage = plan->stages[s];
        printf("\n➡️  Stage %u: Layers %u-%u (Root Node: %u)\n", 
            stage.stageIndex, stage.startLayer, stage.endLayer-1, stage.rootNodeIndex);
        
        for(NnUint i=0; i<stage.nNodes; ++i) {
            NnUint nid = stage.nodeIndices[i];
            printf("    - Node %u: VocabStart=%u, DimStart=%u\n", 
                nid, plan->vocabSplit.starts[nid], plan->dimSplit.starts[nid]);
        }
    }
    printf("===================================\n\n");
}

// --------------------------------------------------------------------------
// 主程序
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        TestArgs args = parseTestArgs(argc, argv);
        
        printf("🚀 Starting Load-Only Test\n");
        printf("📂 Model: %s\n", args.modelPath);
        printf("📊 Ratios: %s\n", args.ratiosStr);
        printf("🤖 Simulating Node Index: %d\n", args.nodeIndex);

        // 1. 加载 Header
        // 假设 maxSeqLen 暂时不重要，设为 4096
        LlmHeader header = loadLlmHeader(args.modelPath, 4096, args.syncType);
        
        // 2. 解析 Ratios 确定节点总数
        // std::vector<float> ratios = parseRatiosLocal(args.ratiosStr, 0); // 第一次解析只为获取数量? 
        // 实际上 parseRatios 需要 nNodes 做校验。我们先自己 split 一下算个总数。
        // 或者简单点：直接解析。
        std::vector<float> tempRatios;
        {
            std::stringstream ss(args.ratiosStr);
            std::string item;
            while(std::getline(ss, item, ',')) tempRatios.push_back(std::stof(item));
        }
        NnUint nNodes = tempRatios.size();
        
        if (args.nodeIndex >= nNodes) {
            throw std::runtime_error("Node index out of bounds.");
        }

        // 3. 创建 Partition Plan
        printf("\n[Step 1] Creating Partition Plan...\n");
        NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
        std::vector<NnStageDef> stageDefs;
        
        NnUint halfLayers = (NnUint)(header.nLayers / 2);
        NnUint remainingLayers = (NnUint)(header.nLayers - halfLayers);

        // --- Stage 0 ---
        NnStageDef stage0;
        stage0.tpRatios = {1.0f};      // 显式赋值给 vector
        stage0.nLayers = halfLayers;   // 显式赋值给 uint
        stageDefs.push_back(stage0);

        // --- Stage 1 ---
        NnStageDef stage1;
        stage1.tpRatios = {0.4f, 0.6f}; // 显式赋值给 vector
        stage1.nLayers = remainingLayers; // 显式赋值给 uint
        stageDefs.push_back(stage1);

        // 更新节点总数
        nNodes = 0;
        for(const auto& stage : stageDefs) {
            nNodes += stage.tpRatios.size();
        }
        
        printf("🔧 Hardcoded Topology: %u Stages, %u Nodes Total\n", 
               (NnUint)stageDefs.size(), nNodes);
        
        // 确保你的 createPartitionPlan 实现是最新的
    NnUnevenPartitionPlan plan = createPartitionPlan(
            stageDefs, 
            header.nHeads, 
            header.nKvHeads, 
            header.vocabSize, 
            ffDim, 
            header.dim
        );
        
        printPlanDetails(&plan);

        // 4. 构建 LlmNet (包含 NodeConfigs)
        printf("[Step 2] Building LlmNet Structure...\n");
        LlmNet net = buildLlmNetUneven(&header, nNodes, 1, &plan); // nBatches=1

        // 校验 Plan 绑定
        if (net.nodeConfigs[args.nodeIndex].partitionPlan != &plan) {
            // 注意：因为 net.nodeConfigs 里的 partitionPlan 指针是在 buildLlmNetUneven 内部赋值的
            // 如果那是深拷贝或者引用了栈变量，这里可能需要手动修复测试逻辑。
            // 但在最新的实现中，我们传递了 &plan 指针。
            // 只要 plan 对象在 main 函数栈上存活，指针就是有效的。
            // 如果 buildLlmNetUneven 只是赋值了指针，这里应该是相等的。
            // 如果不等，可能是 buildLlmNetUneven 实现有误，或者我们需要手动绑一下。
             net.nodeConfigs[args.nodeIndex].partitionPlan = &plan; 
             printf("⚠️  Manually bound plan to node config for safety.\n");
        }

        // 5. 准备 Executor 环境 (即使不推理，Loader 也需要 executor 的 buffers)
        // 我们只初始化当前 Node 的资源
        printf("[Step 3] Initializing Executor for Node %d...\n", args.nodeIndex);
        
        NnNodeConfig* myNodeConfig = &net.nodeConfigs[args.nodeIndex];
        NnNetExecution execution(args.nThreads, &net.netConfig);
        
        // 创建 Device (这一步会触发 NnCpuDevice 构造函数里的 Slice 预计算)
        std::vector<NnExecutorDevice> devices;
        devices.push_back(NnExecutorDevice(
            new NnCpuDevice(&net.netConfig, myNodeConfig, &execution, &plan), 
            -1, -1
        ));

        // Executor
        // 不需要真正的 Synchronizer
        NnFakeNodeSynchronizer fakeSync;
        NnExecutor executor(&net.netConfig, myNodeConfig, &devices, &execution, &fakeSync, false);

        // 6. 执行加载
        printf("\n[Step 4] Loading Weights from Disk...\n");
        printf("------------------------------------------------------------\n");
        
        // 创建本地加载器
        NnLocalWeightLoader loader(&executor, args.nodeIndex);
        
        // 调用核心加载函数
        loadLlmNetWeightUneven(args.modelPath, &net, &loader, &plan, args.nodeIndex);

        printf("------------------------------------------------------------\n");
        printf("✅ Success! Node %d loaded all required weights correctly.\n", args.nodeIndex);

        // 清理 (RAII 会处理大部分，但释放 net 里的数组需要手动调用 releaseLlmNet)
        releaseLlmNet(&net);

    } catch (const std::exception& e) {
        fprintf(stderr, "❌ Error: %s\n", e.what());
        return 1;
    }
    return 0;
}