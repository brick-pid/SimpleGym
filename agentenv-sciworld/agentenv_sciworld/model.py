from pydantic import BaseModel


class ResetRequestBody(BaseModel):
    env_id: int
    task_id: int
