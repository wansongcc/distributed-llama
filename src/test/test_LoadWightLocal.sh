#!/bin/bash

# 配置 (请修改为您的实际路径)
MODEL_PATH="/workspace/distributed-llama/models/qwen3_0.6b_q40/dllama_model_qwen3_0.6b_q40.m"
TOKENIZER_PATH="/workspace/distributed-llama/models/qwen3_0.6b_q40/dllama_tokenizer_qwen3_0.6b_q40.t"
WORKER_PORT=9999
RATIOS="1.0,2.0" # 1:2 比例 (Total 3 nodes? No, ratios length == nodes. Let's say 2 nodes)

# 假设 nNodes = 2 (Root + 1 Worker)
RATIOS="1.0,2.0"

echo "=== 分布式本地加载测试 ==="
echo "清理旧进程..."
pkill -f "dllama worker"
pkill -f "dllama inference"
sleep 1

echo "1. 启动 Worker (Node 1) [本地加载模式]"
echo "   注意: 必须提供 --model 和 --ratios 才能触发本地加载"
# 在后台启动，重定向日志到 worker.log
./dllama worker \
    --port $WORKER_PORT \
    --nthreads 4 \
    --model "$MODEL_PATH" \
    --ratios "$RATIOS" \
    > worker.log 2>&1 &

WORKER_PID=$!
echo "   Worker PID: $WORKER_PID"

# 等待 Worker 启动并开始监听
sleep 2
echo "   检查 Worker 日志..."
head -n 10 worker.log

echo ""
echo "2. 启动 Root (Node 0) [本地加载模式]"
# Root 连接到 Worker
./dllama inference \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --buffer-float-type q80 \
    --prompt "Hello, distributed world!" \
    --steps 16 \
    --nthreads 4 \
    --workers "127.0.0.1:$WORKER_PORT" \
    --ratios "$RATIOS"

# 检查结果
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功！Root 节点完成推理。"
else
    echo "❌ 测试失败！Root 节点退出代码: $EXIT_CODE"
fi

# 清理 Worker
echo "关闭 Worker..."
kill $WORKER_PID