from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEnvWrapper(ABC):
    """Abstract base class that every environment wrapper must implement.

    Each worker process creates one instance via a factory function.
    The wrapper manages multiple environment instances keyed by integer IDs.
    """

    ls: List[int]  # active environment IDs

    @abstractmethod
    def create_with_id(self, idx: int) -> dict:
        """Create an environment with a pre-assigned *idx*."""
        ...

    @abstractmethod
    def step(self, idx: int, action: str) -> dict:
        """Execute *action* in environment *idx* and return observation dict."""
        ...

    @abstractmethod
    def reset(self, idx: int, **kwargs: Any) -> dict:
        """Reset environment *idx*.  Extra keyword arguments are env-specific."""
        ...

    @abstractmethod
    def close(self, idx: int) -> bool:
        """Close environment *idx* and release resources."""
        ...
