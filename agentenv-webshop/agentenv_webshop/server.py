"""
FastAPI Server
"""

import logging
import time

from fastapi import FastAPI, Request

from .utils import register_error_handlers
from .environment import webshop_env_server
from .model import *
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
register_error_handlers(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# 自定义中间件
@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - {response.status_code} - {process_time:.2f} seconds"
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "service": "webshop"}


@app.post("/create")
async def create():
    env = webshop_env_server.create()
    return {"env_id": env}


@app.post("/step")
def step(step_query: StepQuery):
    observation, reward, done, info = webshop_env_server.step(step_query.env_id, step_query.action)
    info = info or {}
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/reset")
def reset(reset_query: ResetQuery):
    result = webshop_env_server.reset(reset_query.env_id, reset_query.task_id)
    # WebShop's env.reset() returns a tuple/list [observation, info]
    # Extract the observation string to match other environments' format
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        observation = result[0]
    else:
        observation = result
    return {"observation": observation, "info": {}}


@app.post("/close")
def close(body: CloseRequestBody):
    result = webshop_env_server.close(body.env_id)
    return {"closed": bool(result), "env_id": body.env_id}
