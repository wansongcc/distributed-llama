#include "nn/nn-core.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>

// -----------------------------------------------------------------------------
// 辅助打印函数：可视化 Plan 结构
// -----------------------------------------------------------------------------
void printPartitionPlanDebug(const NnUnevenPartitionPlan* plan) {
    printf("\n🔍 [DEBUG] Pipeline Partition Plan Verification:\n");
    printf("===================================================\n");
    printf("🌎 Global Stats: Total Nodes: %u, Total Stages: %u\n", plan->nNodes, plan->nStages);

    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig& stage = plan->stages[s];
        printf("\n➡️  [Stage %u]\n", stage.stageIndex);
        printf("    ├─ Range:      Layers %u to %u (Count: %u)\n", 
               stage.startLayer, stage.endLayer - 1, stage.nLayers);
        printf("    ├─ Root Node:  %u\n", stage.rootNodeIndex);
        printf("    ├─ Member Nodes: [ ");
        for(NnUint i=0; i<stage.nNodes; ++i) printf("%u ", stage.nodeIndices[i]);
        printf("]\n");

        printf("    └─ 🔍 TP Split Isolation Check:\n");
        NnUint headSum = 0;
        NnUint kvSum = 0;
        NnUint dimSum = 0;

        for(NnUint i=0; i<stage.nNodes; ++i) {
            NnUint globalNodeIdx = stage.nodeIndices[i];
            
            NnUint hLen = plan->headSplit.lengths[globalNodeIdx];
            NnUint kLen = plan->kvHeadSplit.lengths[globalNodeIdx];
            NnUint dLen = plan->dimSplit.lengths[globalNodeIdx];
            
            headSum += hLen;
            kvSum += kLen;
            dimSum += dLen;

            printf("       • Node %u: Heads=%3u, KV=%3u, Dim=%4u\n", 
                   globalNodeIdx, hLen, kLen, dLen);
        }
        
        printf("       ✅ Stage Sums: Heads=%u, KV=%u, Dim=%u\n", headSum, kvSum, dimSum);
    }
    printf("===================================================\n\n");
}

// -----------------------------------------------------------------------------
// 测试逻辑
// -----------------------------------------------------------------------------
int main() {
    try {
        printf("🧪 Starting Pipeline Parallelism (PP) + Tensor Parallelism (TP) Test...\n");

        // 1. 定义模拟的模型参数 (Qwen 0.6B scale)
        NnUint globalNHeads = 16;
        NnUint globalNKvHeads = 8;
        NnUint globalVocabSize = 151936;
        NnUint globalFfnDim = 3072; // 或者 moeHiddenDim
        NnUint globalDim = 1024;    // Hidden Size

        // 2. 定义 Stage 结构
        std::vector<NnStageDef> stageDefs;

        // --- Stage 0 ---
        // 节点: 0, 1
        // 负责: 前 14 层
        // TP 比例: 1:3 (Node 0 弱, Node 1 强)
        stageDefs.push_back({
            10,                 // nLayers
            {1.0f, 3.0f}        // tpRatios
        });

        // --- Stage 1 ---
        // 节点: 2, 3
        // 负责: 后 14 层
        // TP 比例: 1:1 (Node 2, 3 性能均衡)
        stageDefs.push_back({
            14,                 // nLayers
            {1.0f, 9.0f}        // tpRatios
        });

        // 3. 调用核心切分函数
        // 注意：我们不需要手动计算总节点数，createPartitionPlan 内部会根据 ratios 统计
        NnUnevenPartitionPlan plan = createPartitionPlan(
            stageDefs,
            globalNHeads,
            globalNKvHeads,
            globalVocabSize,
            globalFfnDim,
            globalDim
        );

        // 4. 打印并人工验证
        printPartitionPlanDebug(&plan);

        // 5. 自动断言验证 (Automated Assertions)
        
        // 验证全局属性
        assert(plan.nStages == 2);
        assert(plan.nNodes == 4);

        // 验证 Stage 0 (TP 1:3)
        // HiddenDim 1024 -> Node 0 拿 256, Node 1 拿 768
        assert(plan.dimSplit.lengths[0] == 256);
        assert(plan.dimSplit.lengths[1] == 768);
        // Sum check
        assert(plan.dimSplit.lengths[0] + plan.dimSplit.lengths[1] == globalDim);

        // 验证 Stage 1 (TP 1:1)
        // HiddenDim 1024 -> Node 2 拿 512, Node 3 拿 512
        // 如果 Isolation 没生效，这里可能会变成 256 或其他值
        assert(plan.dimSplit.lengths[2] == 512);
        assert(plan.dimSplit.lengths[3] == 512);
        // Sum check
        assert(plan.dimSplit.lengths[2] + plan.dimSplit.lengths[3] == globalDim);

        // 验证 GQA 对齐
        // Stage 0: KV 1:3 -> Node0(2), Node1(6)
        // Stage 0: Q  1:3 -> Node0(4), Node1(12) -> Ratio 2.0 (Correct)
        assert(plan.headSplit.lengths[0] == 4);
        assert(plan.kvHeadSplit.lengths[0] == 2);
        
        printf("✅ All automated assertions passed!\n");
        printf("✅ Step 1 (Configuration & Topology) is successfully implemented.\n");

    } catch (const std::exception& e) {
        printf("❌ Test Failed with Exception: %s\n", e.what());
        return 1;
    }

    return 0;
}