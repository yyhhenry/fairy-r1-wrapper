# Fairy R1 Wrapper

考虑到Fairy R1模型在返回思维链的时候，因为Chat Template已经把`<think>`当作输入了，在默认的OpenAI like API中不再会输出最初的`<think>`标签，因此我们提供了一个包装器来处理这个问题。

具体而言包装器根据可配置的远程端点URL，作为一个反向代理，代理一切请求，并正确处理SSE；仅当模型是`"fairy-r1"`时，在第一次返回输出的时候，把`<think>`标签加到开头，如果是SSE模式，则应该在开头插入两个数据包以保持格式。

## Test

原本为昇腾卡上部署的Fairy R1模型设计，为了方便Windows调试，用ollama服务示意

```bash
uv run main.py --remote-endpoint http://127.0.0.1:11434/v1 --wrapped-model qwen3:4b

# 列出模型
curl http://127.0.0.1:1075/models

# 尝试流式对话 qwen3，并观察多出来的"<think>", "\n\n"
curl -N -X POST http://127.0.0.1:1075/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3:latest",
  "messages": [
    {"role": "user", "content": "Count from a to d, split by comma.\n/nothink"}
  ],
  "stream": true
}'
```
