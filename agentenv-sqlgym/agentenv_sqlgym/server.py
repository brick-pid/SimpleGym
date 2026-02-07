"""
FastAPI Server
"""

import logging
import time
from typing import List, Literal, Tuple

from fastapi import FastAPI, Request

from .environment import sqlgym_env_server
from .model import *
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
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


@app.get("/", response_model=str)
async def generate_ok():
    """Test connectivity"""
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    """List all environments"""
    return list(sqlgym_env_server.env.keys())


@app.post("/create")
async def create():
    """Create a new environment"""
    env = sqlgym_env_server.create()

    return {"env_id": env}


@app.post("/step", response_model=StepResponse)
async def step(step_query: StepQuery):
    print("/step")
    env_id = step_query.env_id
    print(env_id)
    print(step_query.action)
    state, reward, done, info = sqlgym_env_server.step(
        env_id, step_query.action
    )
    print(env_id)
    print(state)
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/observation", response_model=str)
async def observation(env_id: int):
    print("/observation")
    print(env_id)
    res = sqlgym_env_server.observation(env_id)
    return res


@app.post("/reset", response_model=Tuple[str, None])
async def reset(reset_query: ResetQuery):
    print(reset_query)
    return sqlgym_env_server.reset(reset_query.env_id, reset_query.task_id), None
