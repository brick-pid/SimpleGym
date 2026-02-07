from pydantic import BaseModel


class MazeStepRequestBody(BaseModel):
    env_id: int
    action: str


class MazeResetRequestBody(BaseModel):
    env_id: int
    task_id: int
