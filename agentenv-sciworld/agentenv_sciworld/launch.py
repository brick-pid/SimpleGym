"""
Entrypoint for the SciWorld agent environment.
"""

import os

from agentenv_pool import base_parser, run_server


def launch():
    """entrypoint for `sciworld` command"""

    parser = base_parser(default_parallel_actor=8)
    args = parser.parse_args()

    os.environ["SCIWORLD_PARALLEL_ACTOR"] = str(args.parallel_actor)
    os.environ["SCIWORLD_IPC_TIMEOUT"] = str(args.ipc_timeout)

    run_server(
        "agentenv_sciworld:app",
        host=args.host,
        port=args.port,
    )
