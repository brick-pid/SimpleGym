"""
Mock 并发测试脚本 — 用 processed JSONL 轨迹回放 action 序列

支持环境: alfworld, sciworld, searchqa, webshop

用法:
    python tests/test_mock.py --env-name alfworld --parallel 4
"""

import argparse
import asyncio
import glob
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

import aiohttp


# ── JSONL file patterns per env ──────────────────────────────

JSONL_PATTERNS = {
    "alfworld": "alfworld_*.jsonl",
    "sciworld": "sciworld_*.jsonl",
    "searchqa": "searchqa_*.jsonl",
    "webshop": "webshop_*.jsonl",
}


# ── Per-env reset body builders ──────────────────────────────

def _build_reset_alfworld(env_id: int, task_idx: int) -> dict:
    return {"env_id": env_id, "task_id": task_idx, "world_type": "Text"}


def _build_reset_default(env_id: int, task_idx: int) -> dict:
    return {"env_id": env_id, "task_id": task_idx}


ENV_CONFIGS = {
    "alfworld": {"build_reset": _build_reset_alfworld, "create_needs_task_id": False},
    "sciworld": {"build_reset": _build_reset_default, "create_needs_task_id": False},
    "searchqa": {"build_reset": _build_reset_default, "create_needs_task_id": True},
    "webshop":  {"build_reset": _build_reset_default, "create_needs_task_id": False},
}


# ── Result dataclass ─────────────────────────────────────────

@dataclass
class TrajectoryResult:
    traj_index: int
    task_idx: int
    success: bool
    error: str = ""
    steps_executed: int = 0
    total_actions: int = 0
    early_done: bool = False
    final_reward: float = 0.0
    expected_reward: float = 0.0
    reward_match: bool = False
    create_time: float = 0.0
    reset_time: float = 0.0
    step_times: list = field(default_factory=list)
    close_time: float = 0.0
    # full replay trace for debugging mismatches: list of (action, observation, reward, done)
    replay_trace: list = field(default_factory=list)


# ── Data loading ─────────────────────────────────────────────

def load_trajectories(env_name: str) -> list[dict]:
    traj_dir = os.path.join(os.path.dirname(__file__), "traj", "processed")
    pattern = os.path.join(traj_dir, JSONL_PATTERNS[env_name])
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no JSONL files matching {pattern}")
        sys.exit(1)

    trajs = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trajs.append(json.loads(line))
    print(f"Loaded {len(trajs)} trajectories from {len(files)} file(s)")
    return trajs


# ── Health check ─────────────────────────────────────────────

async def health_check(session: aiohttp.ClientSession, base_url: str) -> bool:
    try:
        async with session.get(f"{base_url}/health") as resp:
            if resp.status != 200:
                print(f"Health check failed: HTTP {resp.status}")
                return False
            data = await resp.json()
            if data.get("status") != "ok":
                print(f"Health check failed: {data}")
                return False
            print(f"Health check OK: {data}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


# ── Single trajectory replay ─────────────────────────────────

async def replay_trajectory(
    session: aiohttp.ClientSession,
    traj: dict,
    traj_index: int,
    base_url: str,
    env_name: str,
) -> TrajectoryResult:
    cfg = ENV_CONFIGS[env_name]
    task_idx = traj["configs"]["task"]["task_idx"]
    expected_reward = traj["configs"].get("reward", 0.0)
    actions = traj.get("actions", [])
    result = TrajectoryResult(
        traj_index=traj_index, task_idx=task_idx,
        success=False, total_actions=len(actions),
        expected_reward=expected_reward,
    )

    env_id = None
    try:
        # create
        t = time.perf_counter()
        create_body = {"task_id": task_idx} if cfg["create_needs_task_id"] else None
        if create_body:
            async with session.post(f"{base_url}/create", json=create_body) as resp:
                resp.raise_for_status()
                data = await resp.json()
        else:
            async with session.post(f"{base_url}/create") as resp:
                resp.raise_for_status()
                data = await resp.json()
        env_id = data["env_id"]
        result.create_time = time.perf_counter() - t

        # reset
        t = time.perf_counter()
        reset_body = cfg["build_reset"](env_id, task_idx)
        async with session.post(f"{base_url}/reset", json=reset_body) as resp:
            resp.raise_for_status()
            await resp.json()
        result.reset_time = time.perf_counter() - t

        # step through actions
        for action in actions:
            t = time.perf_counter()
            async with session.post(
                f"{base_url}/step", json={"env_id": env_id, "action": action}
            ) as resp:
                resp.raise_for_status()
                step_data = await resp.json()
            elapsed = time.perf_counter() - t
            result.step_times.append(elapsed)
            result.steps_executed += 1
            result.final_reward = step_data.get("reward", 0.0)
            result.replay_trace.append({
                "action": action,
                "observation": step_data.get("observation", ""),
                "reward": step_data.get("reward", 0.0),
                "done": step_data.get("done", False),
            })
            if step_data.get("done", False):
                result.early_done = True
                break

        # close
        t = time.perf_counter()
        async with session.post(
            f"{base_url}/close", json={"env_id": env_id}
        ) as resp:
            resp.raise_for_status()
        result.close_time = time.perf_counter() - t
        result.success = True
        result.reward_match = abs(result.final_reward - expected_reward) < 1e-6

    except Exception as e:
        result.error = str(e)
        if env_id is not None:
            try:
                async with session.post(
                    f"{base_url}/close", json={"env_id": env_id}
                ) as _:
                    pass
            except Exception:
                pass

    return result


# ── Concurrent runner ────────────────────────────────────────

async def run_concurrent(
    trajs: list[dict],
    parallel: int,
    base_url: str,
    env_name: str,
    timeout: int,
) -> list[TrajectoryResult]:
    sem = asyncio.Semaphore(parallel)
    connector = aiohttp.TCPConnector(limit=parallel * 2)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async def bounded(session, traj, idx):
        async with sem:
            return await replay_trajectory(session, traj, idx, base_url, env_name)

    async with aiohttp.ClientSession(
        connector=connector, timeout=client_timeout
    ) as session:
        # health check first
        if not await health_check(session, base_url):
            print("Aborting: health check failed")
            sys.exit(1)

        tasks = [bounded(session, t, i) for i, t in enumerate(trajs)]
        results = await asyncio.gather(*tasks)
    return results


# ── Result summary ───────────────────────────────────────────

def _stat(values: list[float]) -> str:
    if not values:
        return "N/A"
    mn = min(values)
    mx = max(values)
    avg = statistics.mean(values)
    med = statistics.median(values)
    return f"min={mn:.3f}s  avg={avg:.3f}s  med={med:.3f}s  max={mx:.3f}s"


def print_summary(results: list[TrajectoryResult], wall_time: float):
    total = len(results)
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    early = [r for r in ok if r.early_done]

    print("\n" + "=" * 60)
    print(f"  RESULTS: {len(ok)}/{total} succeeded, {len(fail)} failed")
    print(f"  Early done (env returned done before all actions): {len(early)}")
    print(f"  Wall time: {wall_time:.2f}s")
    if total > 0:
        print(f"  Throughput: {total / wall_time:.2f} trajs/s")
    print("=" * 60)

    if ok:
        print(f"\n  Create  : {_stat([r.create_time for r in ok])}")
        print(f"  Reset   : {_stat([r.reset_time for r in ok])}")
        all_steps = [t for r in ok for t in r.step_times]
        print(f"  Step    : {_stat(all_steps)}")
        print(f"  Close   : {_stat([r.close_time for r in ok])}")

    # reward correctness
    if ok:
        matched = [r for r in ok if r.reward_match]
        mismatched = [r for r in ok if not r.reward_match]
        print(f"\n  Reward match: {len(matched)}/{len(ok)}")
        if mismatched:
            print(f"  Reward mismatches:")
            for r in mismatched[:10]:
                print(f"    traj#{r.traj_index} task={r.task_idx}: "
                      f"expected={r.expected_reward} got={r.final_reward}")
            if len(mismatched) > 10:
                print(f"    ... and {len(mismatched) - 10} more")
            # print full replay trace for mismatched trajectories
            for r in mismatched:
                print(f"\n  {'─' * 56}")
                print(f"  Replay trace for traj#{r.traj_index} "
                      f"(task={r.task_idx}, {r.steps_executed}/{r.total_actions} steps):")
                for i, step in enumerate(r.replay_trace):
                    print(f"    [{i}] action: {step['action']}")
                    print(f"        obs:    {step['observation'][:200]}")
                    print(f"        reward: {step['reward']}  done: {step['done']}")

    if fail:
        print(f"\n  Failures:")
        for r in fail[:10]:
            print(f"    traj#{r.traj_index} task={r.task_idx}: {r.error}")
        if len(fail) > 10:
            print(f"    ... and {len(fail) - 10} more")


# ── CLI & main ───────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mock concurrent env server test")
    p.add_argument(
        "--env-name", choices=list(JSONL_PATTERNS), default="alfworld",
        help="Environment name (default: alfworld)",
    )
    p.add_argument("--base-url", default=None, help="Server URL (default: from JSONL)")
    p.add_argument("--parallel", type=int, default=8, help="Concurrency (default: 8)")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    p.add_argument("--task-id", type=int, default=None,
                   help="Only run trajectories with this task_idx")
    return p.parse_args()


def main():
    args = parse_args()
    trajs = load_trajectories(args.env_name)
    if not trajs:
        print("No trajectories loaded")
        sys.exit(1)

    if args.task_id is not None:
        trajs = [t for t in trajs if t["configs"]["task"]["task_idx"] == args.task_id]
        if not trajs:
            print(f"No trajectories found with task_idx={args.task_id}")
            sys.exit(1)
        print(f"Filtered to {len(trajs)} trajectory(s) with task_idx={args.task_id}")

    base_url = args.base_url
    if base_url is None:
        base_url = trajs[0]["configs"]["task"]["env_endpoint"]
        print(f"Using base_url from JSONL: {base_url}")

    # strip trailing slash
    base_url = base_url.rstrip("/")

    print(f"Env: {args.env_name}  Parallel: {args.parallel}  "
          f"Trajs: {len(trajs)}  Timeout: {args.timeout}s")
    print(f"Target: {base_url}")

    wall_start = time.perf_counter()
    results = asyncio.run(
        run_concurrent(trajs, args.parallel, base_url, args.env_name, args.timeout)
    )
    wall_time = time.perf_counter() - wall_start

    print_summary(results, wall_time)

    # exit code: 1 if any failures
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
