import os
import json
import threading
from .environment import SingleAlfredTWEnv
from .utils import load_config, process_ob, EnvNotFoundError, EnvClosedError, EpisodeFinishedError, TaskOutOfRangeError, InvalidActionError


class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        # load data_path
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        self._max_id = 0
        self.ls = []
        self.env = {}  # dict[id, env_item]
        self.env_init = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.games = []  # list[game_file]
        self._lock = threading.Lock()       # protects _max_id
        self._tw_lock = threading.Lock()   # protects textworld parser (not thread-safe)
        
        train_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "train"
        )
        test_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_train"
        )

        train_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_train.json",
        )
        test_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_test.json",
        )

        with open(train_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        train_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

        with open(test_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        test_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

    def create(self):
        with self._lock:
            idx = self._max_id
            self._max_id += 1
        self.env[idx] = SingleAlfredTWEnv(self.config)
        self.info[idx] = {"done": False, "reward": 0, "deleted": False}
        print(f"-------Env {idx} created--------")
        self.ls.append(idx)
        return {"env_id": idx}
    
    def __del__(self):
        for idx in self.ls:
            self.env_init[idx].close()
            print(f"-------Env {idx} closed--------")

    def step(self, idx: int, action: str):
        self._check_id(idx)
        with self._tw_lock:
            ob, _, done, info = self.env_init[idx].step([action])
        ob, reward, done = process_ob(ob[0]), float(info["won"][0]), done[0]
        available_actions = info.get("admissible_commands", [[]])[0]
        if ob == "Nothing happens.":
            ob += f"Your action is not valid in current environment. Available action includes {available_actions}."
        payload = {
            "observation": ob,
            "reward": reward,
            "available_actions": available_actions,
            "done": done,
        }
        self.info[idx].update(payload)
        return payload

    def reset(self, idx: int, game: int, world_type: str):
        if world_type not in ["Text", "Embody", "Hybrid"]:
            raise InvalidActionError('world_type must be one of "Text", "Embody" and "Hybrid"')
        if game < 0 or game >= len(self.games):
            raise TaskOutOfRangeError(f"task_id {game} out of range [0, {len(self.games)})")
        self._check_id(idx, True)
        self.env[idx].game_files = [self.games[game]]
        self.env[idx].num_games = 1
        # textworld's tatsu parser is stateful and not thread-safe,
        # so init_env + reset must be serialized.
        with self._tw_lock:
            self.env_init[idx] = self.env[idx].init_env(batch_size=1)
            ob, info = self.env_init[idx].reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        available_actions = info.get("admissible_commands", [[]])[0]
        self.info[idx] = {
            "world_type": world_type,
            "game": game,
            "observation": ob,
            "available_actions": available_actions,
            "done": False,
            "reward": 0,
            "deleted": False,
        }
        return {
            "env_id": idx,
            "observation": ob,
            "available_actions": available_actions,
            "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
        }

    def get_observation(self, idx: int):
        self._check_id(idx)
        return self.info[idx]["observation"]

    def get_available_actions(self, idx: int):
        self._check_id(idx)
        return self.info[idx]["available_actions"]

    def get_detailed_info(self, idx: int):
        self._check_id(idx)
        return self.info[idx]

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise EnvNotFoundError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise EnvClosedError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise EpisodeFinishedError(f"The task with environment {idx} has finished.")

    def close(self, idx: int):
        self._check_id(idx, True)
        try:
            if idx in self.env_init:
                self.env_init[idx].close()
        except Exception:
            pass
        try:
            if idx in self.env:
                self.env[idx].close()
        except Exception:
            pass
        self.info[idx]["deleted"] = True
        if idx in self.ls:
            self.ls.remove(idx)
        return True


os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")
server = ALFWorld_Wrapper(
    data_path=os.environ["ALFWORLD_DATA"],
    config_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "configs", "base_config.yaml"
    ),
)
