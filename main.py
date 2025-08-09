import json
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
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
            "--debug",
            action="store_true",
            help="Enable debug mode",
        )
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


def generate_think_tag(data: dict) -> list[dict]:
    if "choices" not in data or len(data["choices"]) == 0:
        return []

    if "content" not in data["choices"][0]["delta"]:
        return []

    think_data = data.copy()
    think_data["choices"][0]["delta"]["content"] = "<think>"
    think_data["finish_reason"] = None

    newline_data = think_data.copy()
    newline_data["choices"][0]["delta"]["content"] = "\n"
    return [think_data, newline_data]


async def handle_streaming_response(response: httpx.Response, request_body: bytes):
    try:
        request_data = json.loads(request_body.decode())
        model_name = request_data.get("model", "")
        is_wrapped_model = model_name in global_args.wrapped_model
    except (json.JSONDecodeError, UnicodeDecodeError):
        is_wrapped_model = False

    if is_wrapped_model:
        print(f"Processing with wrapped model: {model_name}")

    async def generate_response() -> AsyncGenerator[str, None]:
        think_tag_added = False

        async for line in response.aiter_lines():
            print(repr(line))
            if not line.startswith("data: "):
                yield line + "\n"
                continue

            data_content = line.removeprefix("data: ").strip()

            if data_content == "[DONE]":
                yield line + "\n"
                continue

            try:
                data = json.loads(data_content)

                if is_wrapped_model and not think_tag_added:
                    for think_data in generate_think_tag(data):
                        yield f"data: {json.dumps(think_data)}\n"
                    think_tag_added = True

            except Exception as e:
                print(f"Error processing line: {line}, error: {e}")

            yield line + "\n"

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


async def proxy_request(request: Request, path: str):
    url = f"{global_args.remote_endpoint.rstrip('/')}/{path.lstrip('/')}"

    body = await request.body()

    # Remove unnecessary headers
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
            params=request.query_params,
        ) as response:
            if "text/event-stream" in response.headers.get("content-type", ""):
                return await handle_streaming_response(response, body)
            else:
                return Response(
                    content=await response.aread(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
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
