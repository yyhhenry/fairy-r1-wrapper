import copy
import json
from typing import Optional

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
            default=1075,
            help="Port to listen on",
        )
        parser.add_argument(
            "--remote-endpoint",
            type=str,
            default="http://localhost:1025/v1",
            help="Remote endpoint URL",
        )
        parser.add_argument(
            "--wrapped-model",
            type=str,
            nargs="+",
            default=["FairyR1"],
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
    try:
        assert data["choices"][0]["delta"]["content"]
    except (KeyError, AssertionError):
        return []

    think_data = copy.deepcopy(data)
    think_data["choices"][0]["delta"]["content"] = "<think>"
    think_data["choices"][0]["finish_reason"] = None

    newline_data = copy.deepcopy(think_data)
    newline_data["choices"][0]["delta"]["content"] = "\n\n"
    return [think_data, newline_data]


async def handle_sse(response: httpx.Response, is_wrapped_model: bool):
    think_tag_added = False

    async for line in response.aiter_lines():
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
                    yield f"data: {json.dumps(think_data, separators=(',', ':'))}\n"
                think_tag_added = True

        except Exception as e:
            print(f"Error processing line: {line}, error: {e}")

        yield line + "\n"


class FlexibleResponse(BaseModel):
    status_code: int
    headers: dict[str, str]
    body: bytes | None = None
    stream: bool = False


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
                yield FlexibleResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    stream=True,
                )
                if path == "chat/completions":
                    body_json: dict = json.loads(body)
                    is_wrapped_model = (
                        body_json.get("model") in global_args.wrapped_model
                    )
                    if is_wrapped_model:
                        print(f"Wrapped model detected: {body_json.get('model')}")
                else:
                    is_wrapped_model = False

                async for chunk in handle_sse(response, is_wrapped_model):
                    yield chunk
            else:
                yield FlexibleResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=await response.aread(),
                )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "remote_endpoint": global_args.remote_endpoint}


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
)
async def proxy_all(request: Request, path: str):
    results = proxy_request(request, path)
    flexible_response = await anext(results)
    assert isinstance(flexible_response, FlexibleResponse)
    if flexible_response.stream:
        return StreamingResponse(
            results,
            status_code=flexible_response.status_code,
            headers=flexible_response.headers,
        )
    else:
        return Response(
            flexible_response.body,
            status_code=flexible_response.status_code,
            headers=flexible_response.headers,
        )


def main():
    print("Starting Fairy R1 Wrapper")
    print(f"Proxying to: {global_args.remote_endpoint}")
    print(f"Wrapped models: {global_args.wrapped_model}")

    uvicorn.run(app, host=global_args.host, port=global_args.port)


if __name__ == "__main__":
    main()
