import functools
import logging
import os

from fastapi import FastAPI

from agentenv_pool import Router, create_app, StepRequestBody, CloseRequestBody
from .model import ResetRequestBody
from .env_wrapper import ALFWorld_Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s",
)

_parallel_actor = int(os.environ.get("ALFWORLD_PARALLEL_ACTOR", "64"))
_ipc_timeout = float(os.environ.get("ALFWORLD_IPC_TIMEOUT", "120.0"))
_data_path = os.environ.get(
    "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
)
_config_path = os.environ.get(
    "ALFWORLD_CONFIG",
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..", "configs", "base_config.yaml",
    ),
)


def _make_wrapper():
    return ALFWorld_Wrapper(data_path=_data_path, config_path=_config_path)


router = Router(
    parallel_actor=_parallel_actor,
    wrapper_factory=_make_wrapper,
    ipc_timeout=_ipc_timeout,
)


def _register_routes(application: FastAPI, r: Router):
    @application.post("/create")
    async def create():
        return await r.create()

    @application.post("/step")
    async def step(body: StepRequestBody):
        return await r.step(body.env_id, body.action)

    @application.post("/reset")
    async def reset(body: ResetRequestBody):
        return await r.reset(
            body.env_id, task_id=body.task_id, world_type=body.world_type
        )

    @application.post("/close")
    async def close(body: CloseRequestBody):
        return await r.close(body.env_id)


app = create_app(router, extra_setup=_register_routes)
