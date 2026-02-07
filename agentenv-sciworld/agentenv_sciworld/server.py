from fastapi import FastAPI
import os
from .model import *
from .environment import server

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
    return "This is environment ScienceWorld."


@app.post("/create")
def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.env_id, body.action)

@app.post("/step_visual")
def step_visual(body: StepRequestBody):
    return server.step_visual(body.env_id, body.action)

@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.env_id, body.task_id)

@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.env_id)

@app.get("/observation")
def get_observation(env_id: int):
    return server.get_observation(env_id)


@app.get("/action_hint")
def get_action_hint(env_id: int):
    return server.get_action_hint(env_id)


@app.get("/goals")
def get_goals(env_id: int):
    return server.get_goals(env_id)


@app.get("/detail")
def get_detailed_info(env_id: int):
    return server.get_detailed_info(env_id)


@app.get("/task_description")
def get_task_description(env_id: int):
    return server.get_task_description(env_id)

@app.get("/object_tree")
def get_object_tree(env_id: int):
    return server.get_object_tree(env_id)

@app.get("/state")
def get_current_state(env_id: int):
    return server.get_current_state(env_id)
