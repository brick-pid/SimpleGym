"""IPC protocol definitions for worker pool communication."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class CommandType(Enum):
    CREATE = auto()
    STEP = auto()
    RESET = auto()
    CLOSE = auto()
    SHUTDOWN = auto()
    PING = auto()


@dataclass
class IPCRequest:
    request_id: str
    command: CommandType
    env_id: int = -1
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IPCResponse:
    request_id: str
    success: bool
    payload: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
