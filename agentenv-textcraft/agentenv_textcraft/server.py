from fastapi import FastAPI
import os
from .model import *
from .env_wrapper import server

app = FastAPI()

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def hello():
    return "This is environment TextCraft."


@app.post("/create")
async def create(body: CreateRequestBody):
    return server.create(body.commands, body.goal)


@app.post("/step")
def step(body: StepRequestBody):
    env_id = body.env_id
    print(f"/step {env_id} {body.action}")
    return server.step(env_id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    env_id = body.env_id
    task_id = body.task_id
    print(f"/reset {env_id} {task_id}")
    return server.reset(env_id, task_id)


@app.get("/observation")
def get_observation(env_id: int):
    print(f"/observation {env_id}")
    return server.get_observation(env_id)


@app.get("/commands")
def get_commands(env_id: int):
    return server.get_commands(env_id)


@app.get("/goal")
def get_goal(env_id: int):
    return server.get_goal(env_id)


@app.get("/detail")
def get_detailed_info(env_id: int):
    return server.get_detailed_info(env_id)

@app.post("/close")
def close(body: CloseRequestBody):
    env_id = body.env_id
    print(f"/close {env_id}")
    return server.close(env_id)
