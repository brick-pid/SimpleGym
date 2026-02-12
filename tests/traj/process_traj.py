"""Process raw trajectory data into JSONL format for mock testing."""

import json
import os
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"

ACTION_RE = re.compile(r"<action>(.*?)</action>")
TURN_SPLIT_RE = re.compile(r"^(System|User|Assistant):", re.MULTILINE)


def extract_assistant_actions(text: str) -> list[str]:
    """Extract actions only from Assistant turns, ignoring system prompt templates.

    Within a single Assistant turn, the model may produce multiple retry chunks
    separated by "--" lines (streaming retries). We search from the last chunk
    backwards to find the first chunk that contains a complete <action> tag.
    """
    parts = TURN_SPLIT_RE.split(text)
    actions = []
    for i, part in enumerate(parts):
        if part == "Assistant" and i + 1 < len(parts):
            block = parts[i + 1]
            chunks = re.split(r"\n--\n", block)
            # search from last chunk backwards for a complete action
            for chunk in reversed(chunks):
                found = ACTION_RE.findall(chunk)
                if found:
                    actions.append(found[0])
                    break
    return actions


def process_trajectory(traj_dir: Path) -> dict:
    react_path = traj_dir / "react.txt"
    metrics_path = traj_dir / "metrics.json"

    with open(react_path, "r", encoding="utf-8") as f:
        text = f.read()
    actions = extract_assistant_actions(text)

    with open(metrics_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    return {"actions": actions, "configs": configs}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_dir in sorted(RAW_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        traj_dirs = sorted(
            [d for d in dataset_dir.iterdir() if d.is_dir()]
        )

        records = []
        for traj_dir in traj_dirs:
            try:
                record = process_trajectory(traj_dir)
                records.append(record)
            except Exception as e:
                print(f"WARN: skipping {traj_dir.name}: {e}")

        out_path = OUT_DIR / f"{dataset_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"{dataset_name}: {len(records)} trajectories -> {out_path.name}")


if __name__ == "__main__":
    main()
