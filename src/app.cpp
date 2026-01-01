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

// ---------------------------------------------------------
// Root control-packet logging (default OFF)
// Enable with: -DDLLAMA_CONTROL_LOG=1
// ---------------------------------------------------------
#ifndef DLLAMA_CONTROL_LOG
#define DLLAMA_CONTROL_LOG 0
#endif

static inline void logRootControlSend(const LlmControlPacket& p) {
#if DLLAMA_CONTROL_LOG
    // Match worker-side format for easy diff/align.
    printf("📤 [Root] Send Control: Batch=%u, Pos=%u, Flags=0x%x\n", p.batchSize, p.position, p.flags);
#else
    (void)p;
#endif
}

static void writeBootstrapPacket(NnNetwork *network, NnUint socketIndex, const AppCliArgs *args) {
    LlmBootstrapPacket p;
    p.magic = LLM_BOOTSTRAP_MAGIC;
    p.version = LLM_BOOTSTRAP_VERSION;
    p.flags = 0u;
    p.benchmarkEnabled = args->benchmark ? 1u : 0u;
    p.maxSeqLen = args->maxSeqLen;
    p.syncType = (NnUint)args->syncType;
    p.modelPathLen = 0u;
    p.ratiosLen = 0u;

    if (args->modelPath != nullptr) {
        p.flags |= LLM_BOOTSTRAP_HAS_MODEL_PATH;
        p.modelPathLen = (NnUint)std::strlen(args->modelPath) + 1u;
    }
    if (args->ratiosStr != nullptr) {
        p.flags |= LLM_BOOTSTRAP_HAS_RATIOS;
        p.ratiosLen = (NnUint)std::strlen(args->ratiosStr) + 1u;
    }

    network->write(socketIndex, &p, sizeof(p));
    if (p.modelPathLen > 0u) network->write(socketIndex, args->modelPath, p.modelPathLen);
    if (p.ratiosLen > 0u) network->write(socketIndex, args->ratiosStr, p.ratiosLen);
}

static LlmBootstrapPacket readBootstrapPacket(NnNetwork *network, std::string &modelPath, std::string &ratiosStr) {
    LlmBootstrapPacket p;
    network->read(ROOT_SOCKET_INDEX, &p, sizeof(p));
    if (p.magic != LLM_BOOTSTRAP_MAGIC)
        throw std::runtime_error("Invalid bootstrap magic (root/worker binary mismatch)");
    if (p.version != LLM_BOOTSTRAP_VERSION)
        throw std::runtime_error("Unsupported bootstrap version (root/worker binary mismatch)");

    modelPath.clear();
    ratiosStr.clear();
    if ((p.flags & LLM_BOOTSTRAP_HAS_MODEL_PATH) != 0u) {
        std::vector<char> buf(p.modelPathLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.modelPathLen);
        modelPath.assign(buf.data());
    }
    if ((p.flags & LLM_BOOTSTRAP_HAS_RATIOS) != 0u) {
        std::vector<char> buf(p.ratiosLen);
        network->read(ROOT_SOCKET_INDEX, buf.data(), p.ratiosLen);
        ratiosStr.assign(buf.data());
    }
    return p;
}

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
    args.benchmark = false;
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
    // Parse arguments. Some options are flags (no value), others require a value.
    // NOTE: --workers consumes a variable number of args.
    for (; i < argc; ) {
        char *name = argv[i];

        // Flags (no value)
        if (std::strcmp(name, "--benchmark") == 0) {
            // Support both: "--benchmark" and "--benchmark 1|0".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.benchmark = std::atoi(argv[i + 1]) != 0;
                i += 2;
            } else {
                args.benchmark = true;
                i += 1;
            }
            continue;
        }

        // Options with special arity
        if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;
            if (count <= 0)
                throw std::runtime_error("--workers requires at least one worker in host:port format");

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

            i = j;
            continue;
        }

        // All remaining options require a value
        if (i + 1 >= argc)
            throw std::runtime_error(std::string("Missing value for option: ") + name);
        char *value = argv[i + 1];

        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--ratios") == 0) {
            args.ratiosStr = value;
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

        i += 2;
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

    // ---------------------------------------------------------
    // Ratios string formats (both supported; auto-detected):
    //
    // (A) Legacy per-stage TP ratios (recommended to keep using):
    //   "tp0*tp1*tp2"  where each tp is node ratios (',' or ':' separated)
    //   Optional explicit layers:
    //     - Preferred: append "@<nLayers>" to a stage (works with ':' or ',')
    //     - Legacy:    append ":<nLayers>" ONLY when ratios use commas (e.g. "1,1:10")
    //   Examples:
    //     - 2 nodes, 2 stages: "1*1"
    //     - 4 nodes, 2 stages: "1,1*1,1" or "1:1*1:1"
    //     - explicit layers:   "1:1@10*1:1@18" or "1,1:10*1,1:18"
    //
    // (B) Two-level ratios (stage weights + per-stage TP ratios):
    //   "stageWeights*tpStage0*tpStage1*..."
    //   stageWeights is a list of weights (':' or ',' separated), one per stage.
    //   Each tpStageK is that stage's intra-TP node ratios (':' or ',' separated).
    //   Example (your case): stage weights 1:2; stage0 nodes 1:1; stage1 nodes 2:3
    //     - nNodes=4: "1:2*1:1*2:3"
    // ---------------------------------------------------------

    auto splitStages = [](const std::string& raw) -> std::vector<std::string> {
        std::string s2 = raw;
        for (char &c : s2) {
            if (c == ';' || c == '|') c = '*';
        }
        std::vector<std::string> parts;
        std::stringstream ss2(s2);
        std::string seg;
        while (std::getline(ss2, seg, '*')) {
            if (!seg.empty()) parts.push_back(seg);
        }
        return parts;
    };

    auto isAllDigits = [](const std::string& t) -> bool {
        if (t.empty()) return false;
        for (char c : t) {
            if (c < '0' || c > '9') return false;
        }
        return true;
    };

    // Parse a segment that may be:
    //  - ratios only: "1,1" or "1:1"
    //  - ratios + explicit layers (unambiguous): "1:1@10" or "1,1@10"
    //  - legacy ratios + explicit layers: "1,1:10" (ONLY when ratios use commas)
    //
    // NOTE: We intentionally DO NOT interpret a trailing ":<digits>" as layers when
    // ratios are separated by ':' (e.g. "1:2"), because that would be ambiguous.
    // Returns {ratios, nLayersExplicit(0 if none)}
    auto parseRatiosAndMaybeLayers = [&](const std::string& segment) -> std::pair<std::vector<float>, NnUint> {
        NnUint explicitLayers = 0;
        std::string ratioPart = segment;

        // Preferred, unambiguous layer syntax: "...@<int>"
        {
            const size_t atPos = segment.rfind('@');
            if (atPos != std::string::npos && atPos + 1 < segment.size()) {
                const std::string tail = segment.substr(atPos + 1);
                if (isAllDigits(tail)) {
                    try {
                        explicitLayers = (NnUint)std::stoul(tail);
                        ratioPart = segment.substr(0, atPos);
                    } catch (...) {
                        // ignore
                        explicitLayers = 0;
                        ratioPart = segment;
                    }
                }
            }
        }

        // Legacy layer syntax: "1,1:10" (ONLY when ratios use commas)
        if (explicitLayers == 0u) {
            const bool hasComma = (segment.find(',') != std::string::npos);
            if (hasComma) {
                const size_t lastColon = segment.rfind(':');
                if (lastColon != std::string::npos && lastColon + 1 < segment.size()) {
                    const std::string tail = segment.substr(lastColon + 1);
                    if (isAllDigits(tail)) {
                        try {
                            explicitLayers = (NnUint)std::stoul(tail);
                            ratioPart = segment.substr(0, lastColon);
                        } catch (...) {
                            explicitLayers = 0;
                            ratioPart = segment;
                        }
                    }
                }
            }
        }

        // Now parse ratios: allow both ',' and ':' as separators.
        // We normalize ',' to ':' and then split.
        std::string rp = ratioPart;
        for (char &c : rp) {
            if (c == ',') c = ':';
        }
        std::vector<float> ratios;
        std::stringstream ss4(rp);
        std::string r;
        while (std::getline(ss4, r, ':')) {
            if (r.empty()) continue;
            try {
                ratios.push_back(std::stof(r));
            } catch (...) {
                throw std::invalid_argument("Invalid ratio value: " + r);
            }
        }
        if (ratios.empty()) throw std::invalid_argument("Empty ratio list in segment: " + segment);
        return {ratios, explicitLayers};
    };

    auto sumNodeCounts = [](const std::vector<NnStageDef>& st) -> NnUint {
        NnUint n = 0;
        for (const auto& s : st) n += (NnUint)s.tpRatios.size();
        return n;
    };

    auto autoAssignLayers = [&](std::vector<NnStageDef>& stages, const std::vector<float>& stageWeights) {
        // Collect explicit layers
        NnUint totalExplicitLayers = 0;
        std::vector<int> autoLayerIndices;
        for (size_t i = 0; i < stages.size(); ++i) {
            if (stages[i].nLayers == 0) {
                autoLayerIndices.push_back((int)i);
            } else {
                totalExplicitLayers += stages[i].nLayers;
            }
        }

        if (totalExplicitLayers > nLayers) {
            throw std::invalid_argument("Explicit layers count exceeds total model layers");
        }
        NnUint remainingLayers = nLayers - totalExplicitLayers;

        if (autoLayerIndices.empty()) {
            if (remainingLayers != 0) {
                throw std::invalid_argument("Explicit layers sum does not match total model layers");
            }
            return;
        }

        // Compute weights for auto stages
        float totalWeight = 0.0f;
        std::vector<float> w;
        w.reserve(autoLayerIndices.size());
        for (int idx : autoLayerIndices) {
            float ww = (idx >= 0 && (size_t)idx < stageWeights.size()) ? stageWeights[(size_t)idx] : 0.0f;
            w.push_back(ww);
            totalWeight += ww;
        }

        if (totalWeight <= 1e-6f) {
            // Fallback: uniform
            NnUint base = remainingLayers / (NnUint)autoLayerIndices.size();
            NnUint rem = remainingLayers % (NnUint)autoLayerIndices.size();
            for (size_t i = 0; i < autoLayerIndices.size(); ++i) {
                stages[(size_t)autoLayerIndices[i]].nLayers = base + (i < rem ? 1u : 0u);
            }
            return;
        }

        // Proportional with rounding (last stage gets the remainder)
        NnUint allocatedSoFar = 0;
        for (size_t i = 0; i < autoLayerIndices.size(); ++i) {
            int stageIdx = autoLayerIndices[i];
            NnUint myLayers = 0;
            if (i + 1 == autoLayerIndices.size()) {
                myLayers = remainingLayers - allocatedSoFar;
            } else {
                float ratio = w[i] / totalWeight;
                myLayers = (NnUint)std::round(remainingLayers * ratio);
                if (allocatedSoFar + myLayers > remainingLayers) myLayers = remainingLayers - allocatedSoFar;
            }
            stages[(size_t)stageIdx].nLayers = myLayers;
            allocatedSoFar += myLayers;
            printf("⚖️  [Auto-Split] Stage %d (Weight %.2f): Assigned %u layers\n", stageIdx, w[i], myLayers);
        }
    };

    // ---------- Pass 0: tokenize stage segments ----------
    const std::vector<std::string> parts = splitStages(std::string(ratiosStr));
    if (parts.empty()) throw std::invalid_argument("Ratios string is empty");

    // ---------- Pass 1: try legacy parsing ----------
    {
        std::vector<NnStageDef> stages;
        stages.reserve(parts.size());
        for (const auto& seg : parts) {
            NnStageDef st;
            st.nLayers = 0;
            auto parsed = parseRatiosAndMaybeLayers(seg);
            st.tpRatios = std::move(parsed.first);
            st.nLayers = parsed.second;
            stages.push_back(std::move(st));
        }

        const NnUint totalNodesParsed = sumNodeCounts(stages);
        if (totalNodesParsed == nNodes) {
            // Legacy semantics: stage weight derived from sum(tpRatios)
            std::vector<float> stageWeights;
            stageWeights.reserve(stages.size());
            for (const auto& st : stages) {
                float w = 0.0f;
                for (float r : st.tpRatios) w += r;
                stageWeights.push_back(w);
            }
            autoAssignLayers(stages, stageWeights);
            return stages;
        }
    }

    // ---------- Pass 2: two-level parsing (stageWeights + per-stage tp) ----------
    {
        if (parts.size() < 2) {
            throw std::invalid_argument("Invalid ratios format: not enough segments");
        }

        // First segment = stage weights
        std::vector<float> stageWeights;
        {
            auto parsed = parseRatiosAndMaybeLayers(parts[0]);
            if (parsed.second != 0u) {
                throw std::invalid_argument("Stage-weights segment must not specify layers: " + parts[0]);
            }
            stageWeights = std::move(parsed.first);
        }

        const size_t nStages = stageWeights.size();
        if (nStages == 0) throw std::invalid_argument("Stage weights cannot be empty");
        if (parts.size() != 1 + nStages) {
            std::stringstream msg;
            msg << "Two-level ratios expects 1+" << nStages
                << " segments, but got " << parts.size() << ".\n"
                << "Format: stageWeights*tpStage0*tpStage1*...\n"
            	<< "Example: \"1:2*1:1*2:3\"\n"
                << "Optional explicit layers: tpStage0@10 (e.g. \"1:2*1:1@10*2:3@18\")";
            throw std::invalid_argument(msg.str());
        }

        std::vector<NnStageDef> stages;
        stages.reserve(nStages);
        for (size_t i = 0; i < nStages; ++i) {
            const std::string& seg = parts[1 + i];
            NnStageDef st;
            st.nLayers = 0;
            auto parsed = parseRatiosAndMaybeLayers(seg);
            st.tpRatios = std::move(parsed.first);
            st.nLayers = parsed.second;
            stages.push_back(std::move(st));
        }

        const NnUint totalNodesParsed = sumNodeCounts(stages);
        if (totalNodesParsed != nNodes) {
            std::stringstream msg;
            msg << "Ratios defined " << totalNodesParsed
                << " nodes, but expected " << nNodes << ".\n"
                << "Two-level format example (nNodes=4): \"1:2*1:1*2:3\"\n"
                << "(Stage weights 1:2; stage0 nodes 1:1; stage1 nodes 2:3)\n"
                << "Note: use '@<layers>' if you need explicit layer counts (e.g. \"1:2*1:1@10*2:3@18\").";
            throw std::invalid_argument(msg.str());
        }

        autoAssignLayers(stages, stageWeights);
        return stages;
    }
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

static NnUint getStageIndexForNode(const NnUnevenPartitionPlan* plan, NnUint nodeIndex) {
    if (plan == nullptr || plan->nStages == 0) return 0;
    for (NnUint s = 0; s < plan->nStages; ++s) {
        const NnStageConfig* st = &plan->stages[s];
        for (NnUint i = 0; i < st->nNodes; ++i) {
            if (st->nodeIndices[i] == nodeIndex) return st->stageIndex;
        }
    }
    return 0;
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network, const NnUnevenPartitionPlan* plan, bool profileEnabled) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network;
    this->plan = plan;
    this->profileEnabled = profileEnabled;
    this->controlPacket.flags = profileEnabled ? LLM_CTRL_PROFILE : 0u;
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
    if (network != nullptr) {
        logRootControlSend(controlPacket);
        network->writeAll(&controlPacket, sizeof(LlmControlPacket));
    }
    executor->forward();

    if (!profileEnabled) return;

    // Collect per-node timings for this forward() call.
    lastPerf.clear();
    lastPerf.reserve((network != nullptr ? network->nSockets : 0u) + 1u);

    // Root node (node 0)
    {
        LlmPerfPacket p;
        p.position = controlPacket.position;
        p.batchSize = controlPacket.batchSize;
        p.nodeIndex = 0;
        p.stageIndex = getStageIndexForNode(plan, 0);
        p.execUs = executor->getTotalTime(STEP_EXECUTE_OP);
        p.syncUs = executor->getTotalTime(STEP_SYNC_NODES);
        lastPerf.push_back(p);
    }

    // Worker nodes
    if (network != nullptr && network->nSockets > 0) {
        const NnUint nWorkers = network->nSockets;
        const size_t base = lastPerf.size();
        lastPerf.resize(base + nWorkers);

        std::vector<NnSocketIo> ios(nWorkers);
        for (NnUint i = 0; i < nWorkers; ++i) {
            ios[i].socketIndex = i;
            ios[i].data = &lastPerf[base + i];
            ios[i].size = sizeof(LlmPerfPacket);
        }
        network->readMany(nWorkers, &ios[0]);
    }
}

void RootLlmInference::finish() {
    if (network != nullptr) {
        controlPacket.batchSize = 0;
        // Stop packet: position is not meaningful when batchSize==0.
        // Set to 0 to avoid confusing logs / downstream checks.
        controlPacket.position = 0;
        logRootControlSend(controlPacket);
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
        // Stop packet: position is ignored by design.
        printf("📨 [Worker] Recv Control: Batch=0 (stop)\n");
        isFinished = true;
        return true;
    }
    printf("📨 [Worker] Recv Control: Batch=%u, Pos=%u\n", controlPacket.batchSize, controlPacket.position);
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

        // Bootstrap: send modelPath/ratios/maxSeqLen/syncType to workers so they don't need CLI args.
        for (NnUint nodeIndex = 1; nodeIndex < nNodes; ++nodeIndex) {
            const NnUint socketIndex = nodeIndex - 1;
            writeBootstrapPacket(network, socketIndex, args);
        }

        // 初始化 Synchronizer (传入 Plan)
        synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig, planPtr.get()));

        NnRootConfigWriter configWriter(network);
        configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution, planPtr.get());
    const bool profileEnabled = args->benchmark;
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), profileEnabled);

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

    RootLlmInference inference(&net, &execution, &executor, network, planPtr.get(), profileEnabled);

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

        // Read bootstrap settings from root.
        std::string bootModelPath;
        std::string bootRatios;
        LlmBootstrapPacket boot = readBootstrapPacket(network, bootModelPath, bootRatios);

        const bool hasBootModel = !bootModelPath.empty();
        const bool hasBootRatios = !bootRatios.empty();
        const bool useLocalLoading = hasBootModel && hasBootRatios;
        const NnUint bootMaxSeqLen = boot.maxSeqLen;
        const NnFloatType bootSyncType = (NnFloatType)boot.syncType;
        const bool bootBenchmarkEnabled = boot.benchmarkEnabled != 0u;

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

           // 1. Initialize Plan Pointer (Worker Side)
           std::unique_ptr<NnUnevenPartitionPlan> planPtr;
        
           if (useLocalLoading) {
               // Worker 需要重新加载 Header 和 Plan 以确定加载逻辑和切分
               LlmHeader header = loadLlmHeader((char*)bootModelPath.c_str(), bootMaxSeqLen, bootSyncType);
             
             // [兼容性修复] 自动切换 Q80
             if (header.weightType == F_Q40 && header.syncType != F_Q80) {
                 header.syncType = F_Q80;
             }

               std::vector<NnStageDef> stageDefs = parseStageDefs(bootRatios.c_str(), netConfig.nNodes, header.nLayers);
             NnUint ffDim = (header.archType == QWEN3_MOE) ? header.moeHiddenDim : header.hiddenDim;
             

             planPtr.reset(new NnUnevenPartitionPlan(
                 createPartitionPlan(stageDefs, header.nHeads, header.nKvHeads, header.vocabSize, ffDim, header.dim)
             ));
        }

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution, planPtr.get());
        
        // Initialize Synchronizer with Plan
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig, planPtr.get());
        
        // Benchmark flag is provided by root to keep all nodes consistent.
        // Worker CLI --benchmark is no longer required.
        const bool profileEnabled = bootBenchmarkEnabled;
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, profileEnabled);

        if (useLocalLoading) {
            // [Local Loading Mode]
            printf("🚀 Worker %d: Local Loading Mode from %s\n", nodeConfig.nodeIndex, bootModelPath.c_str());
            
            // Reload header for temporary network construction
            LlmHeader header = loadLlmHeader((char*)bootModelPath.c_str(), bootMaxSeqLen, bootSyncType);
            
            // Build temporary Net for loading context
            // 这里我们需要构建一个临时的 LlmNet 结构，因为 loader 需要 net 指针
            // 关键：确保临时 Net 的 Plan 和 NodeConfig 绑定正确
            LlmNet tempNet = buildLlmNetUneven(&header, netConfig.nNodes, 1, planPtr.get());

            // Execute local loading
            NnLocalWeightLoader localLoader(&executor, nodeConfig.nodeIndex);
            
            // 使用新版 5 参数加载函数
            loadLlmNetWeightUneven((char*)bootModelPath.c_str(), &tempNet, &localLoader, planPtr.get(), nodeConfig.nodeIndex);

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

                // Send per-forward profile packet to root (optional)
                // IMPORTANT: root will block waiting for these packets when profiling is enabled.
                // So workers must reply whenever the control packet requests profiling,
                // even if the worker binary was started without --benchmark (times may be 0).
                if ((inference.flags() & LLM_CTRL_PROFILE) != 0u) {
                    LlmPerfPacket p;
                    p.position = inference.position();
                    p.batchSize = inference.batchSize();
                    p.nodeIndex = nodeConfig.nodeIndex;
                    p.stageIndex = getStageIndexForNode(planPtr.get(), nodeConfig.nodeIndex);
                    p.execUs = executor.getTotalTime(STEP_EXECUTE_OP);
                    p.syncUs = executor.getTotalTime(STEP_SYNC_NODES);
                    network->write(ROOT_SOCKET_INDEX, &p, sizeof(LlmPerfPacket));
                }
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