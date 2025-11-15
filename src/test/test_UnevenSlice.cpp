// test_main.cpp
#include "nn/nn-core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <numeric> // for std::accumulate
#include "llm.hpp"
// 假设 llm.hpp 在这里，但我们只需要它的结构体声明
// 如果没有，您可以模拟一个
#ifndef LLM_HPP 
#define LLM_HPP
enum LlmArchType { LLAMA, QWEN3, QWEN3_MOE };
typedef struct {
    NnUint dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen;
    LlmArchType archType;
    float ropeTheta, normEpsilon;
    NnUint moeHiddenDim;
    // ... 其他 LlmHeader 成员
} LlmHeader;
#endif // LLM_HPP

// --- 您的 nn-core.cpp 中的所有函数声明 ---
// (您必须在此处声明或在 nn-core.hpp 中声明所有新函数)
NnUnevenPartitionPlan createPartitionPlan(
    NnUint nNodes, const std::vector<float>& ratios,
    NnUint globalNHeads, NnUint globalNKvHeads,
    NnUint globalVocabSize, NnUint globalFfnDim
);
void releasePartitionPlan(NnUnevenPartitionPlan* plan);
NnKvCacheSliceUneven sliceKvCacheUneven(NnUint seqLen, NnUint headDim,
                                        const NnUnevenPartitionPlan* plan, NnUint nodeIndex);
// ... (声明所有其他的 Slicer 和 Splitter 函数) ...

/**
 * @brief 辅助函数：计算 NnDimSplit 中的总长度
 */
NnUint sumDimSplit(const NnDimSplit* split, NnUint nNodes) {
    NnUint sum = 0;
    for (NnUint i = 0; i < nNodes; ++i) {
        sum += split->lengths[i];
    }
    return sum;
}

void printDimSplit(const char* title, const NnDimSplit* split, NnUint nNodes, NnUint globalDim) {
    std::cout << "  --- " << title << " (Total: " << globalDim << ") ---" << std::endl;
    // 设置列宽
    std::cout << std::left // 文本左对齐
              << "    " << std::setw(8) << "Node"
              << std::setw(12) << "Start"
              << std::setw(12) << "Length" << std::endl;
    std::cout << "    --------------------------------" << std::endl;
    for (NnUint i = 0; i < nNodes; ++i) {
        std::cout << std::left
                  << "    " << std::setw(8) << i
                  << std::setw(12) << split->starts[i]
                  << std::setw(12) << split->lengths[i] << std::endl;
    }
    std::cout << std::endl; // 增加一个空行以便阅读
}

int main() {
    std::cout << "--- 开始非均匀切分单元测试 ---" << std::endl;
    
    // test_main.cpp -> main()
    std::cout << "Test 1: createPartitionPlan & releasePartitionPlan..." << std::endl;

    // 1. 定义模拟场景
    const NnUint nNodes = 3;
    const std::vector<float> ratios = {2.0f, 3.0f, 5.0f}; // 比例 20%, 30%, 50%
    
    // 模拟 Llama-7B 的维度
    const NnUint globalNHeads = 32;
    const NnUint globalNKvHeads = 32;
    const NnUint globalVocabSize = 32000;
    const NnUint globalFfnDim = 11008;

    // 2. 调用
    NnUnevenPartitionPlan plan = createPartitionPlan(
        nNodes, ratios, globalNHeads, globalNKvHeads, globalVocabSize, globalFfnDim
    );
    assert(plan.nNodes == nNodes);
    std::cout << "\n  [打印切分结果]" << std::endl;
    printDimSplit("Head Split", &plan.headSplit, nNodes, globalNHeads);
    printDimSplit("KV Head Split", &plan.kvHeadSplit, nNodes, globalNKvHeads);
    printDimSplit("Vocab Split", &plan.vocabSplit, nNodes, globalVocabSize);
    printDimSplit("FFN Split", &plan.ffnSplit, nNodes, globalFfnDim);
    // 3. 断言检查：总和
    assert(sumDimSplit(&plan.headSplit, nNodes) == globalNHeads);
    assert(sumDimSplit(&plan.kvHeadSplit, nNodes) == globalNKvHeads);
    assert(sumDimSplit(&plan.vocabSplit, nNodes) == globalVocabSize);
    assert(sumDimSplit(&plan.ffnSplit, nNodes) == globalFfnDim);

    // 4. 断言检查：特定比例 (2:3:5)
    // globalNHeads = 32. 20% = 6.4 (-> 6), 30% = 9.6 (-> 10), 50% = 16
    // 6 + 10 = 16. 32-16 = 16. (符合 round 逻辑)
    assert(plan.headSplit.lengths[0] == 6);  // 20% of 32, rounded
    assert(plan.headSplit.starts[1] == 6);
    assert(plan.headSplit.lengths[1] == 10); // 30% of 32, rounded
    assert(plan.headSplit.starts[2] == 16);
    assert(plan.headSplit.lengths[2] == 16); // 50% of 32 (剩余)

    // globalFfnDim = 11008. 20% = 2201.6 (-> 2202)
    assert(plan.ffnSplit.lengths[0] == 2202);
    assert(plan.ffnSplit.starts[0] == 0);
    assert(plan.ffnSplit.starts[1] == 2202);
    
    // 5. 释放
    releasePartitionPlan(&plan);
    // 确保它被正确置空
    assert(plan.nNodes == 0);
    assert(plan.headSplit.starts == nullptr);

    std::cout << "  [PASSED] Test 1\n";
    
    std::cout << "--- 所有测试通过 ---" << std::endl;
    return 0;
}