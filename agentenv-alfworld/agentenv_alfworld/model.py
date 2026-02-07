from pydantic import BaseModel


class StepRequestBody(BaseModel):
    env_id: int
    action: str


class ResetRequestBody(BaseModel):
    env_id: int
    task_id: int
    world_type: str = "Text"
