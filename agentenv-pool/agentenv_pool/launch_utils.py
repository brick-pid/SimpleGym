"""Helpers for building CLI launchers."""

import argparse

import uvicorn


def base_parser(
    default_port: int = 8000,
    default_host: str = "0.0.0.0",
    default_parallel_actor: int = 64,
    default_ipc_timeout: float = 120.0,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-populated with common flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--host", type=str, default=default_host)
    parser.add_argument(
        "--parallel-actor", type=int, default=default_parallel_actor,
        help="Number of worker subprocesses",
    )
    parser.add_argument(
        "--ipc-timeout", type=float, default=default_ipc_timeout,
        help="Timeout in seconds for IPC calls to workers",
    )
    return parser


def run_server(
    app_import: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start uvicorn with standard settings."""
    uvicorn.run(
        app_import,
        host=host,
        port=port,
        workers=1,
        access_log=False,
    )
