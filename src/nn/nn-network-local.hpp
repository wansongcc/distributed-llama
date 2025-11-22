#ifndef NN_NETWORK_LOCAL_HPP
#define NN_NETWORK_LOCAL_HPP

#include "nn-core.hpp"
#include <functional>

class NnExecutor; // 前置声明

// [分离] 仅用于本地加载权重的加载器
class NnLocalWeightLoader {
public:
    NnLocalWeightLoader(NnExecutor *executor, NnUint nodeIndex);
    ~NnLocalWeightLoader();

    // 基础加载接口
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);

    // 非均匀加载接口
    NnSize loadRowMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                     std::function<NnRowMatmulSliceUneven(NnUint)> slicer, NnByte *weight);
                                     
    NnSize loadColMatmulSlicesUneven(const char *opName, const NnUint opIndex, const NnUint expertIndex, 
                                     std::function<NnColMatmulSliceUneven(NnUint)> slicer, NnByte *weight);

    void finish(); // 空实现

private:
    void allocate(NnSize size);

    NnExecutor *executor;
    NnUint myNodeIndex;
    NnByte *temp;
    NnSize tempSize;
};

#endif