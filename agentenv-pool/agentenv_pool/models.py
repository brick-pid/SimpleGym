from pydantic import BaseModel


class StepRequestBody(BaseModel):
    env_id: int
    action: str


class CloseRequestBody(BaseModel):
    env_id: int
