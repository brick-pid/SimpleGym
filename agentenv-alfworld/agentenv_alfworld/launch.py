"""
Entrypoint for the AlfWorld agent environment.
"""

import os

from agentenv_pool import base_parser, run_server


def launch():
    """entrypoint for `alfworld` commond"""

    parser = base_parser(default_parallel_actor=64)
    args = parser.parse_args()

    os.environ["ALFWORLD_PARALLEL_ACTOR"] = str(args.parallel_actor)
    os.environ["ALFWORLD_IPC_TIMEOUT"] = str(args.ipc_timeout)

    run_server(
        "agentenv_alfworld:app",
        host=args.host,
        port=args.port,
    )

