CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security -Isrc

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifdef DLLAMA_VULKAN
	CGLSLC = glslc

ifeq ($(OS),Windows_NT)
	LIBS += -L$(VK_SDK_PATH)\lib -lvulkan-1
	CXXFLAGS += -DDLLAMA_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DDLLAMA_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

.PHONY: clean dllama

clean:
	$(DELETE_CMD) *.o dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe

# nn
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-network-local.o: src/nn/nn-network-local.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@


ifdef DLLAMA_VULKAN
VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
DEPS += $(VULKAN_SHADER_BINS)

%.spv: %.comp
	$(CGLSLC) -c $< -o $@ --target-env=vulkan1.2
nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o nn-vulkan.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
endif

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-cpu-ops.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
# --- (您的新测试目标) ---
uneven-slice-test: src/test/test_UnevenSlice.cpp nn-quants.o nn-core.o 
# [TAB] <-- 这一行必须以 TAB 开始！
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
worker-load-test: src/test/test_LoadWeightLoacl.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-network-local.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
# [TAB]
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
run-worker-load-test: worker-load-test
# [TAB] 请替换为您的实际路径
	./worker-load-test /workspace/distributed-llama/models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m "1.0,3.0"
dllama: src/dllama.cpp nn-quants.o nn-network-local.o nn-network.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
uneven-llm-build-test: src/test/test_UnevenLlmBuild.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)