# Fairy R1 Wrapper

考虑到Fairy R1模型在返回思维链的时候，因为Chat Template已经把`<think>`当作输入了，在默认的OpenAI like API中不再会输出最初的`<think>`标签，因此我们提供了一个包装器来处理这个问题。

具体而言包装器根据可配置的远程端点URL，作为一个反向代理，代理一切请求，并正确处理SSE；仅当模型是`"fairy-r1"`时，在第一次返回输出的时候，把`"<think>\n"`标签加到开头。
