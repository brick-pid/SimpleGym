from fastapi import FastAPI
from .env_wrapper import server
from .model import *

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment AlfWorld."


@app.post("/create")
async def create():
    return server.create()


@app.post("/step")
async def step(body: StepRequestBody):
    return server.step(body.env_id, body.action)


@app.post("/reset")
async def reset(body: ResetRequestBody):
    print("body", body)
    return server.reset(body.env_id, body.task_id, body.world_type)


@app.get("/available_actions")
def get_available_actions(env_id: int):
    return server.get_available_actions(env_id)


@app.get("/observation")
def get_observation(env_id: int):
    return server.get_observation(env_id)


@app.get("/detail")
def get_detailed_info(env_id: int):
    return server.get_detailed_info(env_id)
