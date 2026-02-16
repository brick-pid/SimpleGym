"""

"""
import os
import json
from .environment import SingleAlfredTWEnv
from agentenv_pool import BaseEnvWrapper
from agentenv_pool.errors import (
    EnvNotFoundError,
    EnvClosedError,
    EpisodeFinishedError,
    TaskOutOfRangeError,
    InvalidActionError,
)
from .utils import load_config

class ALFWorld_Wrapper(BaseEnvWrapper):
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

        self.ls = []
        self.env = {}  # dict[id, env_item]
        self.env_init = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.games = []  # list[game_file]
        
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

    def create_with_id(self, env_id):
        self.env[env_id] = SingleAlfredTWEnv(self.config)
        self.info[env_id] = {"done": False, "reward": 0, "deleted": False}
        print(f"-------Env {env_id} created--------")
        self.ls.append(env_id)
        return {"env_id": env_id}

    def step(self, env_id: int, action: str):
        self._check_id(env_id)
        ob, _, done, info = self.env_init[env_id].step([action])
        ob, reward, done = ob[0], float(info["won"][0]), done[0]
        available_actions = info.get("admissible_commands", [[]])[0]
        if ob == "Nothing happens.":
            ob += f"Your action is not valid in current environment. Available action includes {available_actions}."
        payload = {
            "observation": ob,
            "reward": reward,
            "done": done,
            "info": {"available_actions": available_actions}
        }
        self.info[env_id].update(payload)
        return payload

    def reset(self, env_id: int, task_id: int, world_type: str="Text"):
        if world_type not in ["Text", "Embody", "Hybrid"]:
            raise InvalidActionError('world_type must be one of "Text", "Embody" and "Hybrid"')
        if task_id < 0 or task_id >= len(self.games):
            raise TaskOutOfRangeError(f"task_id {task_id} out of range [0, {len(self.games)})")
        self._check_id(env_id, True)
        self.env[env_id].game_files = [self.games[task_id]]
        self.env[env_id].num_games = 1
        self.env_init[env_id] = self.env[env_id].init_env(batch_size=1)
        ob, info = self.env_init[env_id].reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        available_actions = info.get("admissible_commands", [[]])[0]
        self.info[env_id] = {
            "world_type": world_type,
            "task_id": task_id,
            "observation": ob,
            "available_actions": available_actions,
            "done": False,
            "reward": 0,
            "deleted": False,
        }
        return {
            "observation": ob,
            "info": {
                "env_id": env_id,
                "available_actions": available_actions,
                "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
            }
        }

    def get_observation(self, env_id: int):
        self._check_id(env_id)
        return self.info[env_id]["observation"]

    def get_available_actions(self, env_id: int):
        self._check_id(env_id)
        return self.info[env_id]["available_actions"]

    def get_detailed_info(self, env_id: int):
        self._check_id(env_id)
        return self.info[env_id]

    def _check_id(self, env_id: int, is_reset: bool = False):
        if env_id not in self.info:
            raise EnvNotFoundError(f"The id {env_id} is not valid.")
        if self.info[env_id]["deleted"]:
            raise EnvClosedError(f"The task with environment {env_id} has been deleted.")
        if not is_reset and self.info[env_id]["done"]:
            raise EpisodeFinishedError(f"The task with environment {env_id} has finished.")

    def close(self, env_id: int):
        self._check_id(env_id, True)
        try:
            if env_id in self.env_init:
                self.env_init[env_id].close()
        except Exception:
            pass
        try:
            if env_id in self.env:
                self.env[env_id].close()
        except Exception:
            pass
        self.info[env_id]["deleted"] = True
        if env_id in self.ls:
            self.ls.remove(env_id)
        return True
