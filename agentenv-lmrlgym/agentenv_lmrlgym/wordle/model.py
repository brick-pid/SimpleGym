from pydantic import BaseModel


class WordleStepRequestBody(BaseModel):
    env_id: int
    action: str


class WordleResetRequestBody(BaseModel):
    env_id: int
    task_id: int


class WordleCloseRequestBody(BaseModel):
    env_id: int
