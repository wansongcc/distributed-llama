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
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);
    LlmNet net;
    std::unique_ptr<NnUnevenPartitionPlan> planPtr;
    std::vector<float> ratios;

    if(args->ratiosStr != nullptr){
        printf("nNodes=%d\n", nNodes);
        ratios = parseRatios(args->ratiosStr, nNodes);
        NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
        planPtr.reset(new NnUnevenPartitionPlan(
            createPartitionPlan(nNodes, ratios, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim)
        ));
        net = buildLlmNetUneven(&header, nNodes, args->nBatches, ratios);
        if (args->info) {
            printf("⚖️  Uneven partitioning strategy enabled: %s\n", args->ratiosStr);
        }
    }else{
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
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get()));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    // initialize uneven partition plan if needed


    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    // Load weights
    if (args->ratiosStr != nullptr) {
        // [非均匀模式]：强制使用本地加载 (Local Loading)
        // 摒弃网络传输，Root 节点直接从本地文件加载属于自己的部分。
        printf("🚀 Local Loading Mode (Root): Loading weights locally...\n");
        // 创建本地加载器 (指定 Root 的 index为 0)
        NnLocalWeightLoader localLoader(&executor, 0); 
        
        // 调用非均匀加载函数 (传入 Plan 和 LocalLoader)
        loadLlmNetWeightUneven(args->modelPath, &net, &localLoader, planPtr.get());
        printf("✅ Root: Weights loaded locally.\n");

    } else {
        // [均匀模式]：保持原有行为 (使用 NnRootWeightLoader)
        // 这里的 NnRootWeightLoader 可能会通过网络将权重分发给 Worker
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
        
        // Use custom deleters for C-style configs if they need specific cleanup functions
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

        // 1. Initialize Plan Pointer
        std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        
        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
             LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
             std::vector<float> ratios = parseRatios(args->ratiosStr, netConfig.nNodes);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
             
             // Create the plan value
             NnUnevenPartitionPlan plan = createPartitionPlan(
                 netConfig.nNodes, ratios, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim
             );

             // Move it to the heap-managed unique_ptr
             // ideally NnUnevenPartitionPlan should have a move constructor to steal pointers
             planPtr.reset(new NnUnevenPartitionPlan(std::move(plan)));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution);
        
        // Pass raw pointer to synchronizer (it usually just reads from it)
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig, planPtr.get());
        
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        if (args->ratiosStr != nullptr && args->modelPath != nullptr) {
            // [Local Loading Mode]
            printf("🚀 Worker %d: Local Loading Mode from %s\n", nodeConfig.nodeIndex, args->modelPath);
            
            // Reload header for temporary network construction
            LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
            std::vector<float> ratios = parseRatios(args->ratiosStr, netConfig.nNodes);
            
            // Fix dimensions if necessary
            if (header.headDim == 0 && header.nHeads > 0) header.headDim = header.dim / header.nHeads;
            header.qDim = header.nHeads * header.headDim;
            header.kvDim = header.nKvHeads * header.headDim;

            // Build temporary Net for loading
            LlmNet tempNet = buildLlmNetUneven(&header, netConfig.nNodes, 1, ratios);

            // Execute local loading
            NnLocalWeightLoader localLoader(&executor, nodeConfig.nodeIndex);
            
            // Pass the plan pointer
            loadLlmNetWeightUneven(args->modelPath, &tempNet, &localLoader, planPtr.get());

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