#include "app.hpp"
#include <cassert>
#include <cstring>
#include <sstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>
#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif

// 引入 LLM 头文件以获取 createPartitionPlan 等定义
#include "llm.hpp"

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.ratiosStr = nullptr;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // First see if any of the args are asking for help/usage and fail fast
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i += count - 1;
        } else if (strcmp(argv[i], "--ratios") == 0) {
            if (++i >= argc) throw std::runtime_error("--ratios requires an argument (e.g., \"1.0,3.0\")");
            args.ratiosStr = argv[i];
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

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

// [修改] 解析多 Stage 格式: "1.0:10;0.5,0.5:14"
// 1. 分号 ';' 或 竖线 '|' 分隔不同的 Stage
// 2. 冒号 ':' 分隔 TP比例 和 层数 (可选，如果不填则自动分配)
// 3. 逗号 ',' 分隔同一 Stage 内的 TP 节点比例
// [修改] 解析多 Stage 格式，并支持按算力比例自动切分层数
static std::vector<NnStageDef> parseStageDefs(const char *ratiosStr, NnUint nNodes, NnUint nLayers) {
    printf("🔍 [DEBUG] parseStageDefs received: \"%s\"\n", ratiosStr);

    std::vector<NnStageDef> stages;
    std::string s(ratiosStr);

    // 1) 支持多种 Stage 分隔符：'*'、';'、'|' 均可
    // 将所有可能的分隔符统一替换为 '*'
    for (char &c : s) {
        if (c == ';' || c == '|') c = '*';
    }

    std::stringstream ss(s);
    std::string segment;

    NnUint totalExplicitLayers = 0;
    std::vector<int> autoLayerIndices; // 记录哪些 Stage 需要自动分配

    // 2) 解析每个 Stage
    while (std::getline(ss, segment, '*')) {
        if (segment.empty()) continue;

        NnStageDef stage;
        stage.nLayers = 0; // 0 表示未指定，稍后自动分配

        // 检查冒号 ':' (显式指定层数)
        size_t colonPos = segment.find(':');
        std::string ratioPart = segment;

        if (colonPos != std::string::npos) {
            ratioPart = segment.substr(0, colonPos);
            std::string layerPart = segment.substr(colonPos + 1);
            try {
                stage.nLayers = (NnUint)std::stoi(layerPart);
                totalExplicitLayers += stage.nLayers;
            } catch (...) {
                throw std::invalid_argument("Invalid layer count: " + layerPart);
            }
        }

        // 解析 TP 比例（逗号分隔）
        std::stringstream ss2(ratioPart);
        std::string ratio;
        while (std::getline(ss2, ratio, ',')) {
            if (ratio.empty()) continue;
            try {
                stage.tpRatios.push_back(std::stof(ratio));
            } catch (...) {
                throw std::invalid_argument("Invalid ratio value: " + ratio);
            }
        }

        if (stage.tpRatios.empty()) {
            throw std::invalid_argument("Empty stage definition found");
        }

        if (stage.nLayers == 0) {
            autoLayerIndices.push_back((int)stages.size());
        }

        stages.push_back(stage);
    }

    // 3. 校验节点总数
    NnUint totalNodesParsed = 0;
    for(const auto& stage : stages) {
        totalNodesParsed += stage.tpRatios.size();
    }
    
    if (totalNodesParsed != nNodes) {
        // 构造更友好的错误信息，解释 Ratio 语义并给出示例
        std::stringstream msg;
        msg << "Ratios defined " << totalNodesParsed
            << " nodes, but expected " << nNodes << ".\n"
            << "• Commas (,) define TP nodes inside one stage.\n"
            << "• Stages are separated by '*', ';' or '|'.\n"
            << "• Total nodes across ALL stages must equal nNodes (root + workers).\n"
            << "Examples:\n"
            << "  - nNodes=2, two stages: \"1.0*1.0\"\n"
            << "  - nNodes=4, two stages: \"1.0,1.0*1.0,1.0\"\n"
            << "Hint: adjust --workers or ratio string accordingly.";
        throw std::invalid_argument(msg.str());
    }

    // 4. [核心修改] 按权重比例分配层数
    if (totalExplicitLayers > nLayers) {
        throw std::invalid_argument("Explicit layers count exceeds total model layers");
    }

    NnUint remainingLayers = nLayers - totalExplicitLayers;
    size_t nAutoStages = autoLayerIndices.size();

    if (nAutoStages > 0) {
        // 计算每个自动分配 Stage 的“总算力权重”
        std::vector<float> stageWeights;
        float totalWeight = 0.0f;

        for (int idx : autoLayerIndices) {
            float w = 0.0f;
            // Stage 的权重 = 该 Stage 内所有节点的 Ratio 之和
            // (例如 "0.5,0.5" 的权重是 1.0, "2.0" 的权重是 2.0)
            for (float r : stages[idx].tpRatios) w += r;
            stageWeights.push_back(w);
            totalWeight += w;
        }

        if (totalWeight <= 1e-6) {
            // 兜底：如果权重全是 0，退化为平均分配
            NnUint base = remainingLayers / nAutoStages;
            NnUint remain = remainingLayers % nAutoStages;
            for (size_t i = 0; i < nAutoStages; ++i) {
                stages[autoLayerIndices[i]].nLayers = base + (i < remain ? 1 : 0);
            }
        } else {
            // 按比例分配
            NnUint allocatedSoFar = 0;
            for (size_t i = 0; i < nAutoStages; ++i) {
                int stageIdx = autoLayerIndices[i];
                NnUint myLayers;

                if (i == nAutoStages - 1) {
                    // 最后一个 Stage 拿走剩余所有，消除舍入误差
                    myLayers = remainingLayers - allocatedSoFar;
                } else {
                    // 计算比例: (MyWeight / TotalWeight) * Remaining
                    float ratio = stageWeights[i] / totalWeight;
                    myLayers = (NnUint)std::round(remainingLayers * ratio);
                    
                    // 边界检查：防止溢出或分配为0（除非算力真的极小）
                    if (allocatedSoFar + myLayers > remainingLayers) {
                        myLayers = remainingLayers - allocatedSoFar;
                    }
                }
                
                stages[stageIdx].nLayers = myLayers;
                allocatedSoFar += myLayers;
                
                printf("⚖️  [Auto-Split] Stage %d (Weight %.2f): Assigned %u layers\n", 
                       stageIdx, stageWeights[i], myLayers);
            }
        }
    } else {
        if (remainingLayers != 0) {
            throw std::invalid_argument("Explicit layers sum does not match total model layers");
        }
    }
    
    return stages;
}

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

            printf("       • Node %u: Heads=%u, KV=%u, Dim=%u\n", 
                   globalNodeIdx, hLen, kLen, dLen);
        }
        printf("       ✅ Stage Sums: Heads=%u, KV=%u, Dim=%u\n", headSum, kvSum, dimSum);
    }
    printf("===================================================\n\n");
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution, const NnUnevenPartitionPlan *plan = nullptr) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution, plan), -1, -1));
    }
    return devices;
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network; // May be nullptr!
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket.batchSize = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    controlPacket.position = position;
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::forward() {
    if (network != nullptr) 
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    executor->forward();
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    }
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->positionPipe = (float *)execution->pipes[0];
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &controlPacket, sizeof(LlmControlPacket), maxAttempts))
        return false;
    printf("📨 [Worker] Recv Control: Batch=%u, Pos=%u\n", controlPacket.batchSize, controlPacket.position);    
    if (controlPacket.batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    for (NnUint i = 0; i < controlPacket.batchSize; i++)
        positionPipe[i] = (float)(controlPacket.position + i);
    execution->setBatchSize(controlPacket.batchSize);
    return true;
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;
    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);

    if (nNodes > header.nKvHeads)
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);
    LlmNet net;
    std::unique_ptr<NnUnevenPartitionPlan> planPtr;
    std::vector<float> ratios;

    if(args->ratiosStr != nullptr){
        printf("nNodes=%d\n", nNodes);
        std::vector<NnStageDef> stageDefs = parseStageDefs(args->ratiosStr, nNodes, header.nLayers);
        NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;

        planPtr.reset(new NnUnevenPartitionPlan(
            createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim)
        ));
        
        // 使用 Uneven Builder (传入 planPtr)
        net = buildLlmNetUneven(&header, nNodes, args->nBatches, planPtr.get());
        
        if (args->info) {
            printf("⚖️  Uneven partitioning strategy enabled: %s\n", args->ratiosStr);
            printPartitionPlanDebug(planPtr.get());
        }
    } else {
        printf("⚖️  Even partitioning strategy enabled: ");
        net = buildLlmNet(&header, nNodes, args->nBatches);
    }
    
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        // 初始化 Synchronizer (传入 Plan)
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get()));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    // Load weights
    if (args->ratiosStr != nullptr) {
        // [非均匀/PP 模式]：强制使用本地加载 (Local Loading)
        printf("🚀 Local Loading Mode (Root): Loading weights locally...\n");
        
        NnLocalWeightLoader localLoader(&executor, 0); 
        // 传入 0 作为 Root 的 nodeIndex
        loadLlmNetWeightUneven(args->modelPath, &net, &localLoader, planPtr.get(), 0);
        printf("✅ Root: Weights loaded locally.\n");

    } else {
        // [均匀模式]：保持原有行为 (网络分发)
        NnRootWeightLoader weightLoader(&executor, network, nNodes);
        loadLlmNetWeight(args->modelPath, &net, &weightLoader);
    }

    RootLlmInference inference(&net, &execution, &executor, network);

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;

    handler(&context);

    inference.finish();
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

        // 1. Initialize Plan Pointer (Worker Side)
        std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        
        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
             // Worker 需要重新加载 Header 和 Plan 以确定加载逻辑和切分
             LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
             
             // [兼容性修复] 自动切换 Q80
             if (header.weightType == F_Q40 && header.syncType != F_Q80) {
                 header.syncType = F_Q80;
             }

             std::vector<NnStageDef> stageDefs = parseStageDefs(args->ratiosStr, netConfig.nNodes, header.nLayers);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
             

             planPtr.reset(new NnUnevenPartitionPlan(
                 createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim)
             ));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution, planPtr.get());
        
        // Initialize Synchronizer with Plan
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig, planPtr.get());
        
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
            // [Local Loading Mode]
            printf("🚀 Worker %d: Local Loading Mode from %s\n", nodeConfig.nodeIndex, args->modelPath);
            
            // Reload header for temporary network construction
            LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
            
            // Build temporary Net for loading context
            // 这里我们需要构建一个临时的 LlmNet 结构，因为 loader 需要 net 指针
            // 关键：确保临时 Net 的 Plan 和 NodeConfig 绑定正确
            LlmNet tempNet = buildLlmNetUneven(&header, netConfig.nNodes, 1, planPtr.get());

            // Execute local loading
            NnLocalWeightLoader localLoader(&executor, nodeConfig.nodeIndex);
            
            // 使用新版 5 参数加载函数
            loadLlmNetWeightUneven(args->modelPath, &tempNet, &localLoader, planPtr.get(), nodeConfig.nodeIndex);

            releaseLlmNet(&tempNet);
            printf("✅ Worker %d: Weights loaded locally.\n", nodeConfig.nodeIndex);

        } else {
            // [Network Loading Mode] (Legacy)
            printf("📡 Worker %d: Waiting for weights from Root...\n", nodeConfig.nodeIndex);
            NnWorkerWeightReader weightReader(&executor, network);
            weightReader.read();
        }

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
                break;
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
                break;
            }
        }
    }
}