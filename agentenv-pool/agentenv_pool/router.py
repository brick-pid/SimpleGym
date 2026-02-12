"""Router / Manager: manages a pool of worker subprocesses."""

import asyncio
import logging
import multiprocessing
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict

from .ipc import CommandType, IPCRequest, IPCResponse
from .protocol import BaseEnvWrapper
from .errors import (
    EnvError,
    EnvNotFoundError,
    EnvNotReadyError,
    EnvClosedError,
    EpisodeFinishedError,
    TaskOutOfRangeError,
    InvalidActionError,
    ConfigMissingError,
)
from .worker import worker_main

logger = logging.getLogger(__name__)

_ERROR_MAP: Dict[str, type] = {
    "ENV_NOT_FOUND": EnvNotFoundError,
    "ENV_NOT_READY": EnvNotReadyError,
    "ENV_CLOSED": EnvClosedError,
    "EPISODE_FINISHED": EpisodeFinishedError,
    "TASK_OUT_OF_RANGE": TaskOutOfRangeError,
    "INVALID_ACTION": InvalidActionError,
    "CONFIG_MISSING": ConfigMissingError,
}


@dataclass
class WorkerHandle:
    process: multiprocessing.Process
    pipe: Connection
    lock: asyncio.Lock


class Router:
    """Generic worker-pool router.

    Parameters
    ----------
    parallel_actor : int
        Number of worker subprocesses.
    wrapper_factory : Callable[[], BaseEnvWrapper]
        A picklable callable that creates a fresh wrapper instance.
    ipc_timeout : float
        Seconds to wait for a worker response before raising.
    """

    def __init__(
        self,
        parallel_actor: int,
        wrapper_factory: Callable[[], BaseEnvWrapper],
        ipc_timeout: float = 120.0,
    ):
        self._parallel_actor = parallel_actor
        self._wrapper_factory = wrapper_factory
        self._ipc_timeout = ipc_timeout
        self._next_id = 0
        self._id_lock = asyncio.Lock()
        self._workers: Dict[int, WorkerHandle] = {}
        self._executor = ThreadPoolExecutor(max_workers=parallel_actor)

    # ── lifecycle ──────────────────────────────────────────────

    def start_workers(self) -> None:
        for wid in range(self._parallel_actor):
            parent_conn, child_conn = multiprocessing.Pipe()
            p = multiprocessing.Process(
                target=worker_main,
                args=(child_conn, wid, self._parallel_actor,
                      self._wrapper_factory),
                daemon=True,
            )
            p.start()
            child_conn.close()

            if not parent_conn.poll(120):
                p.kill()
                raise RuntimeError(
                    f"Worker {wid} did not become ready in 120s"
                )
            resp: IPCResponse = parent_conn.recv()
            if not resp.success:
                p.kill()
                raise RuntimeError(
                    f"Worker {wid} init failed: {resp.error_message}"
                )

            self._workers[wid] = WorkerHandle(
                process=p, pipe=parent_conn, lock=asyncio.Lock(),
            )
            logger.info("Worker %d started (pid=%d)", wid, p.pid)

        logger.info("All %d workers ready", self._parallel_actor)

    async def shutdown(self) -> None:
        for wid, handle in self._workers.items():
            try:
                req = IPCRequest(
                    request_id=str(uuid.uuid4()),
                    command=CommandType.SHUTDOWN,
                )
                handle.pipe.send(req)
            except Exception:
                logger.warning("Failed to send SHUTDOWN to worker %d", wid)

        for wid, handle in self._workers.items():
            handle.process.join(timeout=10)
            if handle.process.is_alive():
                logger.warning("Worker %d did not exit, killing", wid)
                handle.process.kill()
                handle.process.join(timeout=5)
            handle.pipe.close()

        self._executor.shutdown(wait=False)
        self._workers.clear()
        logger.info("All workers shut down")

    # ── routing ────────────────────────────────────────────────

    def _route(self, env_id: int) -> int:
        return env_id % self._parallel_actor

    # ── IPC ────────────────────────────────────────────────────

    async def _send_to_worker(
        self, worker_id: int, req: IPCRequest
    ) -> IPCResponse:
        handle = self._workers.get(worker_id)
        if handle is None or not handle.process.is_alive():
            raise EnvNotReadyError(
                f"Worker {worker_id} is not available"
            )

        loop = asyncio.get_running_loop()
        async with handle.lock:
            await loop.run_in_executor(
                self._executor, handle.pipe.send, req
            )
            ready = await loop.run_in_executor(
                self._executor, handle.pipe.poll, self._ipc_timeout
            )
            if not ready:
                raise EnvNotReadyError(
                    f"Worker {worker_id} timed out "
                    f"after {self._ipc_timeout}s"
                )
            resp: IPCResponse = await loop.run_in_executor(
                self._executor, handle.pipe.recv
            )
        return resp

    @staticmethod
    def _raise_if_error(resp: IPCResponse) -> Any:
        if resp.success:
            return resp.payload
        cls = _ERROR_MAP.get(resp.error_code, EnvError)
        raise cls(resp.error_message or "Unknown error")

    # ── public API ─────────────────────────────────────────────

    async def create(self) -> dict:
        async with self._id_lock:
            env_id = self._next_id
            self._next_id += 1

        worker_id = self._route(env_id)
        req = IPCRequest(
            request_id=str(uuid.uuid4()),
            command=CommandType.CREATE,
            env_id=env_id,
        )
        resp = await self._send_to_worker(worker_id, req)
        return self._raise_if_error(resp)

    async def step(self, env_id: int, action: str) -> dict:
        worker_id = self._route(env_id)
        req = IPCRequest(
            request_id=str(uuid.uuid4()),
            command=CommandType.STEP,
            env_id=env_id,
            action=action,
        )
        resp = await self._send_to_worker(worker_id, req)
        return self._raise_if_error(resp)

    async def reset(self, env_id: int, **kwargs: Any) -> dict:
        worker_id = self._route(env_id)
        req = IPCRequest(
            request_id=str(uuid.uuid4()),
            command=CommandType.RESET,
            env_id=env_id,
            params=kwargs,
        )
        resp = await self._send_to_worker(worker_id, req)
        return self._raise_if_error(resp)

    async def close(self, env_id: int) -> bool:
        worker_id = self._route(env_id)
        req = IPCRequest(
            request_id=str(uuid.uuid4()),
            command=CommandType.CLOSE,
            env_id=env_id,
        )
        resp = await self._send_to_worker(worker_id, req)
        return self._raise_if_error(resp)
