#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "nn/nn-network-local.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#ifndef DLLAMA_DEBUG_TOPK_LOGITS
#define DLLAMA_DEBUG_TOPK_LOGITS 0
#endif

#if DLLAMA_DEBUG_TOPK_LOGITS
static void printEscapedPiece(const char* s, size_t maxLen = 32) {
    if (s == nullptr) {
        printf("~");
        return;
    }
    for (size_t i = 0; s[i] != '\0' && i < maxLen; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\n') { printf("\\n"); continue; }
        if (c == '\r') { printf("\\r"); continue; }
        if (c == '\t') { printf("\\t"); continue; }
        if (c < 32 || c >= 127) {
            printf("\\x%02x", (unsigned)c);
            continue;
        }
        putchar((int)c);
    }
}

static void debugTopKLogits(const AppInferenceContext* context, const float* logits, NnUint vocabSize, int k, const char* tag) {
    if (context == nullptr || context->tokenizer == nullptr || context->tokenizer->vocab == nullptr) return;
    if (k <= 0) return;
    if ((NnUint)k > vocabSize) k = (int)vocabSize;

    struct Item { float v; int i; };
    std::vector<Item> top;
    top.reserve((size_t)k);

    for (NnUint i = 0; i < vocabSize; ++i) {
        float v = logits[i];
        if ((int)top.size() < k) {
            top.push_back(Item{v, (int)i});
            if ((int)top.size() == k) {
                std::sort(top.begin(), top.end(), [](const Item& a, const Item& b) { return a.v > b.v; });
            }
            continue;
        }
        if (v <= top.back().v) continue;
        int lo = 0;
        int hi = k;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v > top[mid].v) hi = mid; else lo = mid + 1;
        }
        for (int j = k - 1; j > lo; --j) top[j] = top[j - 1];
        top[lo] = Item{v, (int)i};
    }

    printf("🧭 [TopK] %s k=%d\n", tag == nullptr ? "" : tag, k);
    for (int j = 0; j < (int)top.size(); ++j) {
        int id = top[j].i;
        const char* piece = (id >= 0 && (NnUint)id < context->tokenizer->vocabSize) ? context->tokenizer->vocab[id] : nullptr;
        printf("  #%d id=%d logit=%+.4f piece=\"", j, id, top[j].v);
        printEscapedPiece(piece);
        printf("\"\n");
    }
}

static void debugVocabCoverage(const float* logits, NnUint vocabSize, const char* tag) {
    if (logits == nullptr || vocabSize == 0) return;
    const int blocks = 16;
    NnUint blockSize = (vocabSize + blocks - 1) / blocks;
    NnUint zeroCount = 0;
    NnUint nearZeroCount = 0;
    for (NnUint i = 0; i < vocabSize; ++i) {
        float v = logits[i];
        if (v == 0.0f) zeroCount++;
        if (std::fabs(v) < 1e-6f) nearZeroCount++;
    }
    printf("🧱 [VocabCoverage] %s vocab=%u zero=%u (%.1f%%) | |v|<1e-6=%u (%.1f%%)\n",
        tag == nullptr ? "" : tag,
        vocabSize,
        zeroCount,
        vocabSize ? (100.0f * (float)zeroCount / (float)vocabSize) : 0.0f,
        nearZeroCount,
        vocabSize ? (100.0f * (float)nearZeroCount / (float)vocabSize) : 0.0f);

    for (int b = 0; b < blocks; ++b) {
        NnUint lo = (NnUint)b * blockSize;
        NnUint hi = std::min(vocabSize, lo + blockSize);
        if (lo >= hi) break;
        float mx = -1e30f;
        int mxIdx = -1;
        for (NnUint i = lo; i < hi; ++i) {
            float v = logits[i];
            if (v > mx) { mx = v; mxIdx = (int)i; }
        }
        printf("  block[%2d] [%6u..%6u) max=%+.4f idx=%d\n", b, lo, hi, mx, mxIdx);
    }
}
#endif

static void inference(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

#if DLLAMA_DEBUG_TOPK_LOGITS
    if (context->tokenizer->vocabSize != context->header->vocabSize) {
        printf("⚠️ Tokenizer vocabSize=%u != Model vocabSize=%u (强烈怀疑 tokenizer/model 不匹配)\n",
            context->tokenizer->vocabSize,
            context->header->vocabSize);
    }
    {
        int dumpN = std::min(nInputTokens, 32);
        printf("🧾 Prompt tokens n=%d: ", nInputTokens);
        for (int i = 0; i < dumpN; ++i) {
            int id = inputTokens[i];
            const char* piece = (id >= 0 && (NnUint)id < context->tokenizer->vocabSize) ? context->tokenizer->vocab[id] : nullptr;
            printf("%d(\"", id);
            printEscapedPiece(piece, 16);
            printf("\")%s", (i + 1 < dumpN) ? " " : "");
        }
        if (dumpN < nInputTokens) printf(" ...");
        printf("\n");
    }
#endif

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnSize sentBytes = 0;
    NnSize recvBytes = 0;
    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;

    // Per-stage/per-node profiling stats (enabled when executor benchmark is on)
    struct NodePerfAgg {
        unsigned long long execUs = 0;
        unsigned long long syncUs = 0;
        unsigned long long forwardCount = 0;
        unsigned long long tokenCount = 0;
        NnUint stageIndex = 0;
        bool hasStage = false;
    };
    const NnUint nNodes = (context->args->nWorkers + 1);
    std::vector<NodePerfAgg> perfAgg;
    if (context->args->benchmark) {
        perfAgg.resize(nNodes);
    }

    int token = inputTokens[pos];
    printf("%s\n", context->args->prompt);
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        if (context->args->benchmark) {
            const std::vector<LlmPerfPacket>& perf = context->inference->getLastPerf();
            for (const LlmPerfPacket& p : perf) {
                if (p.nodeIndex >= perfAgg.size()) continue;
                NodePerfAgg& a = perfAgg[p.nodeIndex];
                a.execUs += p.execUs;
                a.syncUs += p.syncUs;
                a.forwardCount += 1;
                a.tokenCount += (unsigned long long)std::max<NnUint>(1u, p.batchSize);
                a.stageIndex = p.stageIndex;
                a.hasStage = true;
            }
        }
        float* logits = context->inference->logitsPipe;
        NnUint vocabSize = context->header->vocabSize;
        bool hasNaN = false;
        bool hasInf = false;
        float maxLogit = -1e9;
        float minLogit = 1e9;
        int maxIndex = -1;

        // 简单抽样检查或全量检查
        for (NnUint i = 0; i < vocabSize; ++i) {
            float val = logits[i];
            if (std::isnan(val)) hasNaN = true;
            if (std::isinf(val)) hasInf = true;
            if (val > maxLogit) { maxLogit = val; maxIndex = i; }
            if (val < minLogit) minLogit = val;
        }

#if DLLAMA_DEBUG_TOPK_LOGITS
        // 只在 eval 阶段前几步打印，避免刷屏
        if (pos < 4) {
            debugTopKLogits(context, logits, vocabSize, 10, "eval");
            debugVocabCoverage(logits, vocabSize, "eval");
        }
#endif


        pos += batchSize;
        // 注意：这里是在“评估 prompt（不含最后一个 token）”的循环中，不能用 pos+1，
        // 否则在 pos==nInputTokens-1 时会越界，导致后续生成从错误 token 开始。

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔷️ Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | (%d tokens)\n",
            evalTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            batchSize);
        printf("🧪 [Root Logits] Valid: %s | Range: [%.2f, %.2f] | MaxIdx: %d | NetDelta: S=%zu R=%zu\n", 
            (hasNaN || hasInf) ? "❌ FAIL" : "✅ OK", 
            minLogit, maxLogit, maxIndex, 
            sentBytes, recvBytes);
        evalTotalTime += evalTime + syncTime;
    }

    // 生成阶段的起始 token 应该是 prompt 的最后一个 token（位置为 nInputTokens-1）
    token = inputTokens[nInputTokens - 1];

    fflush(stdout);

    context->inference->setBatchSize(1);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
    for (; pos < maxPos; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        if (context->args->benchmark) {
            const std::vector<LlmPerfPacket>& perf = context->inference->getLastPerf();
            for (const LlmPerfPacket& p : perf) {
                if (p.nodeIndex >= perfAgg.size()) continue;
                NodePerfAgg& a = perfAgg[p.nodeIndex];
                a.execUs += p.execUs;
                a.syncUs += p.syncUs;
                a.forwardCount += 1;
                a.tokenCount += (unsigned long long)std::max<NnUint>(1u, p.batchSize);
                a.stageIndex = p.stageIndex;
                a.hasStage = true;
            }
        }

#if DLLAMA_DEBUG_TOPK_LOGITS
        // 预测阶段：前若干步打印 topK，快速判断 logits 是否“像正常语言模型”
        if (pos < 16) {
            debugTopKLogits(context, context->inference->logitsPipe, context->header->vocabSize, 10, "pred");
            if (pos < 4) debugVocabCoverage(context->inference->logitsPipe, context->header->vocabSize, "pred");
        }
#endif
        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔶 Pred%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | %s\n",
            predTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            piece == nullptr ? "~" : piece);
        fflush(stdout);
        predTotalTime += predTime + syncTime;
    }

    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nEvalTokens * 1000) / evalTotalTimeMs,
        evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nPredTokens * 1000) / predTotalTimeMs,
        predTotalTimeMs / ((float) nPredTokens));

    if (context->args->benchmark && !perfAgg.empty()) {
        printf("\n");
        printf("⏱️  [Stage/Node Profile Summary]\n");
        for (NnUint node = 0; node < (NnUint)perfAgg.size(); ++node) {
            const NodePerfAgg& a = perfAgg[node];
            if (a.forwardCount == 0 || a.tokenCount == 0) continue;

            const double execPerFwdMs = (double)a.execUs / 1000.0 / (double)a.forwardCount;
            const double syncPerFwdMs = (double)a.syncUs / 1000.0 / (double)a.forwardCount;
            const double totalPerFwdMs = execPerFwdMs + syncPerFwdMs;

            const double execPerTokMs = (double)a.execUs / 1000.0 / (double)a.tokenCount;
            const double syncPerTokMs = (double)a.syncUs / 1000.0 / (double)a.tokenCount;
            const double totalPerTokMs = execPerTokMs + syncPerTokMs;

            printf("  • Stage %u Node %u: per-fwd total=%6.2f ms (exec=%6.2f sync=%6.2f) | per-tok total=%6.2f ms (exec=%6.2f sync=%6.2f) | fwd=%llu tok=%llu\n",
                a.hasStage ? a.stageIndex : 0u,
                node,
                totalPerFwdMs,
                execPerFwdMs,
                syncPerFwdMs,
                totalPerTokMs,
                execPerTokMs,
                syncPerTokMs,
                (unsigned long long)a.forwardCount,
                (unsigned long long)a.tokenCount);
        }
        printf("\n");
        printf("Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.\n");
    }
}

static NnUint readStdin(const char *guide, char *buffer, NnUint size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        NnUint length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void perplexity(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    printf("Evaluating %d tokens...\n", nInputTokens);

    float totalLogProb = 0.0f;
    NnUint pos = 0;

    context->inference->setBatchSize(1);

    for (pos = 0; pos < nInputTokens - 1; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, inputTokens[pos]);
        context->inference->forward();

        float *logits = context->inference->logitsPipe;
        softmax_F32(logits, context->header->vocabSize);

        int targetToken = inputTokens[pos + 1];
        float prob = logits[targetToken];

        totalLogProb += std::log(std::max(prob, 1e-30f));
        printf("%5d / %d, prob=%f\n", pos + 1, nInputTokens - 1, prob);
    }

    float avgLogProb = totalLogProb / (float)(nInputTokens - 1);
    float perplexity = expf(-avgLogProb);

    printf("\n");
    printf("Results\n");
    printf("   perplexity: %f (lower = better)\n", perplexity);
    printf("   avgLogProb: %f\n", avgLogProb);
    printf("   bitPerToken: %f\n", -avgLogProb / std::log(2.0));
}

static void chat(AppInferenceContext *context) {
    const NnUint seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);

    const NnUint sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnUint pos = 0;
    NnUint userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        GeneratedChat inputPrompt = templateGenerator.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.length + 2]);
        int *inputTokens = inputTokensPtr.get();

        bool isStart = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.content, inputTokens, &nInputTokens, isStart, true);

        NnUint userPromptEndPos = (NnUint)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        NnUint i = 0;
        for (;;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnUint batchSize = remainingTokens < context->args->nBatches
                ? remainingTokens
                : context->args->nBatches;

            context->inference->setBatchSize(batchSize);
            context->inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                context->inference->setToken(j, inputTokens[i + j]);

            context->inference->forward();

            i += batchSize;
            pos += batchSize;
            // 这里同 inference()：prompt eval 只跑到最后一个 token 的前一位，
            // 循环结束后再用 i 指向的那个“最后 token”启动生成。
        }

        // 生成阶段起始 token：prompt 的最后一个 token
        if (i < (NnUint)nInputTokens) token = inputTokens[i];
        else token = inputTokens[nInputTokens - 1];

        context->inference->setBatchSize(1);
        context->tokenizer->resetDecoder();

        printf("\n🤖 Assistant\n");
        if (inputPrompt.publicPrompt != nullptr)
            printf("%s", inputPrompt.publicPrompt);

        while (pos < seqLen) {
            context->inference->setPosition(pos);
            context->inference->setToken(0, token);
            context->inference->forward();

            token = context->sampler->sample(context->inference->logitsPipe);

            char *piece = context->tokenizer->decode(token);
            EosDetectorType eosType = eosDetector.append(token, piece);
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != nullptr) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

int main(int argc, char **argv) {
    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0) {
            printf("nNodes=%d\n", args.nWorkers);
            runInferenceApp(&args, &inference);
        } else if (std::strcmp(args.mode, "perplexity") == 0)
            runInferenceApp(&args, &perplexity);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}
