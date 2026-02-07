from typing import Any, List, Optional

from pydantic import BaseModel


class StepQuery(BaseModel):
    env_id: int
    action: str


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: Any


class ResetQuery(BaseModel):
    env_id: int
    task_id: int


class CloseRequestBody(BaseModel):
    env_id: int
