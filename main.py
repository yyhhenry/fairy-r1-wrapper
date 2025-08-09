import json
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Fairy R1 Wrapper", description="Fairy R1模型API包装器")


class Args(BaseModel):
    host: str
    port: int
    remote_endpoint: str
    wrapped_model: list[str]

    @classmethod
    def parse_args(cls):
        import argparse

        parser = argparse.ArgumentParser(description="Fairy R1 Wrapper")
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="Port to listen on",
        )
        parser.add_argument(
            "--remote-endpoint",
            type=str,
            default="http://localhost:1025",
            help="Remote endpoint URL",
        )
        parser.add_argument(
            "--wrapped-model",
            type=str,
            nargs="+",
            default=["fairy-r1"],
            help="List of wrapped model names",
        )

        args = parser.parse_args()
        return cls.model_validate(vars(args))


global_args = Args.parse_args()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list] = None


async def handle_streaming_response(
    response: httpx.Response, request_body: bytes
) -> StreamingResponse:
    try:
        request_data = json.loads(request_body.decode())
        model_name = request_data.get("model", "")
        is_wrapped_model = model_name in global_args.wrapped_model
    except (json.JSONDecodeError, UnicodeDecodeError):
        is_wrapped_model = False

    async def generate_response() -> AsyncGenerator[str, None]:
        first_content_chunk = True

        async for chunk in response.aiter_text():
            if not chunk.strip():
                yield chunk
                continue

            lines = chunk.split("\n")
            processed_lines = []

            for line in lines:
                if line.startswith("data: "):
                    data_content = line.removeprefix("data: ").strip()

                    if data_content == "[DONE]":
                        processed_lines.append(line)
                        continue

                    try:
                        data = json.loads(data_content)

                        if (
                            is_wrapped_model
                            and first_content_chunk
                            and "choices" in data
                            and data["choices"]
                        ):
                            choice = data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content and content.strip():
                                    choice["delta"]["content"] = "<think>\n" + content
                                    first_content_chunk = False
                            elif "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                if content and content.strip():
                                    choice["message"]["content"] = "<think>\n" + content
                                    first_content_chunk = False

                        processed_lines.append("data: " + json.dumps(data))
                    except json.JSONDecodeError:
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)

            yield "\n".join(processed_lines)

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


async def proxy_request(request: Request, path: str) -> Response:
    """代理请求到远程端点"""
    url = f"{global_args.remote_endpoint.rstrip('/')}/{path.lstrip('/')}"

    # 获取请求体
    body = await request.body()

    # 构建请求头，排除一些不需要的头
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
                params=request.query_params,
                timeout=60.0,
            )

            # 对于流式响应的特殊处理
            if "text/event-stream" in response.headers.get("content-type", ""):
                return await handle_streaming_response(response, body)
            else:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502, detail=f"Proxy request failed: {str(e)}"
            )


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
)
async def proxy_all(request: Request, path: str):
    return await proxy_request(request, path)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "remote_endpoint": global_args.remote_endpoint}


def main():
    print("Starting Fairy R1 Wrapper")
    print(f"Proxying to: {global_args.remote_endpoint}")
    print(f"Wrapped models: {global_args.wrapped_model}")

    uvicorn.run(app, host=global_args.host, port=global_args.port)


if __name__ == "__main__":
    main()
