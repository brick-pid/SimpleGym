from pydantic import BaseModel
from typing import Optional


class CreateRequestBody(BaseModel):
    commands: Optional[str] = None
    goal: Optional[str] = None


class StepRequestBody(BaseModel):
    env_id: int
    action: str


class ResetRequestBody(BaseModel):
    env_id: int
    task_id: Optional[int] = 0

class CloseRequestBody(BaseModel):
    env_id: int
