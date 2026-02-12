# Re-export error classes from agentenv_pool for backwards compatibility
from agentenv_pool.errors import (  # noqa: F401
    EnvError,
    EnvNotReadyError,
    EnvClosedError,
    EpisodeFinishedError,
    TaskOutOfRangeError,
    InvalidActionError,
    ConfigMissingError,
    EnvNotFoundError,
    register_error_handlers,
)
