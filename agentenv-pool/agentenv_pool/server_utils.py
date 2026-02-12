"""Helpers for building FastAPI applications with the worker pool."""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from starlette.responses import Response

from .errors import register_error_handlers
from .router import Router

logger = logging.getLogger(__name__)


def _add_log_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        body = await request.body()
        response: Response = await call_next(request)
        process_time = time.time() - start_time
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "body": body.decode("utf-8", errors="replace") if body else None,
            "status_code": response.status_code,
            "process_time": f"{process_time:.4f}s",
        }
        logger.info(json.dumps(log_data))
        return response


def create_app(
    router: Router,
    extra_setup: Callable[[FastAPI, Router], None] = None,
) -> FastAPI:
    """Create a FastAPI app wired to the given *router*.

    The returned app includes:
    - lifespan that starts / shuts down workers
    - error handlers
    - request-logging middleware
    - ``/health`` endpoint

    *extra_setup* is an optional callback ``(app, router) -> None`` that can
    register additional routes or middleware.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        router.start_workers()
        yield
        await router.shutdown()

    app = FastAPI(lifespan=lifespan)
    register_error_handlers(app)
    _add_log_middleware(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if extra_setup is not None:
        extra_setup(app, router)

    return app
