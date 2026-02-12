"""Generic worker subprocess for environment instances."""

import logging
import traceback
from typing import Callable
from multiprocessing.connection import Connection

from .ipc import CommandType, IPCRequest, IPCResponse
from .protocol import BaseEnvWrapper
from .errors import EnvError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger(__name__)


def _handle_request(wrapper: BaseEnvWrapper, req: IPCRequest) -> IPCResponse:
    """Dispatch a single IPC request to the wrapper."""
    try:
        if req.command == CommandType.CREATE:
            payload = wrapper.create_with_id(req.env_id)
            return IPCResponse(req.request_id, success=True, payload=payload)

        if req.command == CommandType.STEP:
            payload = wrapper.step(req.env_id, req.action)
            return IPCResponse(req.request_id, success=True, payload=payload)

        if req.command == CommandType.RESET:
            payload = wrapper.reset(req.env_id, **req.params)
            return IPCResponse(req.request_id, success=True, payload=payload)

        if req.command == CommandType.CLOSE:
            payload = wrapper.close(req.env_id)
            return IPCResponse(req.request_id, success=True, payload=payload)

        if req.command == CommandType.PING:
            return IPCResponse(req.request_id, success=True, payload="pong")

        return IPCResponse(
            req.request_id, success=False,
            error_code="INTERNAL_ERROR",
            error_message=f"Unknown command: {req.command}",
        )
    except EnvError as e:
        return IPCResponse(
            req.request_id, success=False,
            error_code=e.code,
            error_message=e.message,
            retryable=e.retryable,
        )
    except Exception as e:
        logger.error("Unhandled error: %s\n%s", e, traceback.format_exc())
        return IPCResponse(
            req.request_id, success=False,
            error_code="INTERNAL_ERROR",
            error_message=str(e),
        )


def worker_main(
    pipe: Connection,
    worker_id: int,
    parallel_actor: int,
    wrapper_factory: Callable[[], BaseEnvWrapper],
):
    """Entry point for a worker subprocess.

    *wrapper_factory* must be a picklable callable (module-level function or
    functools.partial) that returns a fresh BaseEnvWrapper instance.
    """
    logger.info("Worker %d starting (parallel_actor=%d)", worker_id, parallel_actor)

    try:
        wrapper = wrapper_factory()
    except Exception as e:
        logger.error("Worker %d failed to init: %s", worker_id, e)
        pipe.send(IPCResponse("__init__", success=False, error_message=str(e)))
        return

    pipe.send(IPCResponse("__init__", success=True))
    logger.info("Worker %d ready", worker_id)

    while True:
        try:
            req: IPCRequest = pipe.recv()
        except (EOFError, OSError):
            logger.info("Worker %d pipe closed, exiting", worker_id)
            break

        if req.command == CommandType.SHUTDOWN:
            logger.info("Worker %d shutting down", worker_id)
            for idx in list(wrapper.ls):
                try:
                    wrapper.close(idx)
                except Exception:
                    pass
            pipe.send(IPCResponse(req.request_id, success=True))
            break

        resp = _handle_request(wrapper, req)
        try:
            pipe.send(resp)
        except (OSError, BrokenPipeError):
            logger.error("Worker %d cannot send response, exiting", worker_id)
            break

    logger.info("Worker %d exited", worker_id)
