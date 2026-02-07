from typing import List, Optional

from pydantic import BaseModel

class CreateQuery(BaseModel):
    task_id: int = 0


class StepQuery(BaseModel):
    env_id: int
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: None


class ResetQuery(BaseModel):
    env_id: int
    task_id: int = 0


class Task(BaseModel):
    id: str
    content: str
    
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    
class CloseRequestBody(BaseModel):
    env_id: int
