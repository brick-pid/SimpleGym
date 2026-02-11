from fastapi import FastAPI, Request
from starlette.responses import Response
import logging
import time
import json

from .utils import register_error_handlers
from .env_wrapper import server
from .model import *

app = FastAPI()
register_error_handlers(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    params = ""
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes) if body_bytes else {}
            parts = []
            if "env_id" in body_json:
                parts.append(f"env_id={body_json['env_id']}")
            if "task_id" in body_json:
                parts.append(f"task_id={body_json['task_id']}")
            if "action" in body_json:
                action = str(body_json["action"])
                if len(action) > 100:
                    action = action[:100] + "..."
                parts.append(f"action={action}")
            if parts:
                params = " " + " ".join(parts)
        except Exception:
            pass
    response = await call_next(request)
    duration = time.time() - start_time
    error_info = ""
    if response.status_code >= 400:
        try:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk if isinstance(chunk, bytes) else chunk.encode()
            err = json.loads(resp_body)
            if "error" in err:
                e = err["error"]
                error_info = f' error={e.get("code", "")} msg="{e.get("message", "")}"'
            response = Response(content=resp_body, status_code=response.status_code, headers=dict(response.headers), media_type=response.media_type)
        except Exception:
            pass
    logging.info(f'{request.client.host} - "{request.method} {request.url.path}"{params} {response.status_code} {duration:.3f}s{error_info}')
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": "alfworld"}


@app.post("/create")
async def create():
    result = server.create()
    return result


@app.post("/step")
def step(body: StepRequestBody):
    result = server.step(body.env_id, body.action)
    if isinstance(result, dict) and "done" in result:
        return {
            "observation": result.get("observation"),
            "reward": result.get("reward", 0),
            "done": result.get("done", False),
            "info": {
                k: v for k, v in result.items() if k not in {"observation", "reward", "done"}
            },
        }
    return result


@app.post("/reset")
def reset(body: ResetRequestBody):
    result = server.reset(body.env_id, body.task_id, body.world_type)
    if isinstance(result, dict) and "observation" in result:
        return {
            "observation": result.get("observation"),
            "info": {k: v for k, v in result.items() if k != "observation"},
        }
    return {"observation": result, "info": {}}


@app.post("/close")
def close(body: CloseRequestBody):
    result = server.close(body.env_id)
    return {"closed": bool(result), "env_id": body.env_id}
