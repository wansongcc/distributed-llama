#ifndef APP_HPP
#define APP_HPP

#include <vector>
#include <chrono>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network-local.hpp"
#include "tokenizer.hpp"
#include "llm.hpp"

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    NnFloatType syncType;
    NnUint nWorkers;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    char *ratiosStr; 

    // worker
    NnUint port;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    ~AppCliArgs();

};

typedef struct {
    NnUint position;
    NnUint batchSize; // 0 = stop signal
    NnUint flags;     // bit0: enable per-token profiling
} LlmControlPacket;

enum LlmControlFlags : NnUint {
    LLM_CTRL_PROFILE = 1u << 0,
};

typedef struct {
    NnUint position;
    NnUint batchSize;
    NnUint nodeIndex;
    NnUint stageIndex;
    NnUint execUs;
    NnUint syncUs;
} LlmPerfPacket;

// Bootstrap settings sent from root to worker after socket connect and before
// sending net/node configs. This removes the need for workers to pass --model/--ratios.
enum LlmBootstrapFlags : NnUint {
    LLM_BOOTSTRAP_HAS_MODEL_PATH = 1u << 0,
    LLM_BOOTSTRAP_HAS_RATIOS     = 1u << 1,
};

typedef struct {
    NnUint magic;      // 'DLBM'
    NnUint version;    // 2
    NnUint flags;      // LlmBootstrapFlags
    NnUint benchmarkEnabled; // 0/1, enables executor timer on workers
    NnUint maxSeqLen;  // forwarded from root args
    NnUint syncType;   // NnFloatType (as NnUint)
    NnUint modelPathLen; // bytes including '\0' if present
    NnUint ratiosLen;    // bytes including '\0' if present
} LlmBootstrapPacket;

static constexpr NnUint LLM_BOOTSTRAP_MAGIC = 0x4d424c44u; // 'DLBM' little-endian
static constexpr NnUint LLM_BOOTSTRAP_VERSION = 2u;

class RootLlmInference {
public:
    float *logitsPipe;
    const std::vector<LlmPerfPacket>& getLastPerf() const { return lastPerf; }
private:
    float *tokenPipe;
    float *positionPipe;
    LlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    NnNetwork *network;
    LlmControlPacket controlPacket;
    bool profileEnabled = false;
    const NnUnevenPartitionPlan* plan = nullptr;
    std::vector<LlmPerfPacket> lastPerf;
public:
    RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, const NnUnevenPartitionPlan* plan, bool profileEnabled);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
    void finish();
};

class WorkerLlmInference {
public:
    bool isFinished;
    NnUint position() const { return controlPacket.position; }
    NnUint batchSize() const { return controlPacket.batchSize; }
    NnUint flags() const { return controlPacket.flags; }
private:
    float *positionPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    LlmControlPacket controlPacket;
public:
    WorkerLlmInference(NnNetExecution *execution, NnNetwork *network);
    bool tryReadControlPacket();
};

typedef struct {
    AppCliArgs *args;
    LlmHeader *header;
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    NnExecutor *executor;
} AppInferenceContext;

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context));
void runWorkerApp(AppCliArgs *args);

#endif
