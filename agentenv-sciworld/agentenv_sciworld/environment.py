from scienceworld import ScienceWorldEnv

from agentenv_pool import BaseEnvWrapper
from agentenv_pool.errors import (
    EnvNotFoundError,
    EnvClosedError,
    EpisodeFinishedError,
)


class SciWorldWrapper(BaseEnvWrapper):
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.games = []
        self.ls = []
        exceptions = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
        init_env = ScienceWorldEnv()
        for key, value in init_env.tasks.items():
            if key not in exceptions:
                self.games += [
                    {"taskName": value, "variationIdx": i}
                    for i in range(init_env.get_max_variations(value))
                ]
        init_env.close()
        del init_env

    def create_with_id(self, idx: int):
        env = ScienceWorldEnv()
        self.env[idx] = env
        self.info[idx] = {"deleted": False, "done": False}
        self.ls.append(idx)
        print(f"-------Env {idx} created--------")
        return {"env_id": idx}

    def step(self, idx: int, action: str):
        self._check_id(idx)
        if "task_name" not in self.info[idx]:
            raise EpisodeFinishedError(
                f"Environment {idx} has not been reset. "
                "Please call reset before step."
            )
        ob, reward, done, info = self.env[idx].step(action)
        payload = {
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "done": done,
        }
        self.info[idx].update(payload)
        return payload

    def step_visual(self, idx: int, action: str):
        self._check_id(idx)
        processed_action = action
        if processed_action.endswith("</s>"):
            processed_action = processed_action[:-4]
        if "Action:" in processed_action:
            action_parts = processed_action.split("Action:")
            if len(action_parts) > 1:
                processed_action = action_parts[1].strip()
            else:
                processed_action = action_parts[0].strip()
        ob, reward, done, info = self.env[idx].step(processed_action)
        try:
            object_tree = self.env[idx].get_object_tree()
        except Exception:
            object_tree = None
        try:
            inventory = self.env[idx].inventory()
        except Exception:
            inventory = ""
        payload = {
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "done": done,
            "info": info,
            "object_tree": object_tree,
            "inventory": inventory,
            "moves": info.get("moves", 0),
        }
        self.info[idx].update(payload)
        return payload

    def reset(self, idx: int, data_idx=None):
        if data_idx is None:
            data_idx = 0
        self._check_id(idx, True)
        self.env[idx].load(
            self.games[data_idx]["taskName"],
            self.games[data_idx]["variationIdx"],
        )
        task_description = self.env[idx].get_task_description()
        ob, reward, done, info = self.env[idx].step("look around")
        payload = {
            "task_name": self.games[data_idx]["taskName"],
            "var_num": self.games[data_idx]["variationIdx"],
            "task_description": task_description,
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "deleted": False,
            "done": done,
        }
        self.info[idx].update(payload)
        return payload

    def get_observation(self, idx: int):
        self._check_id(idx)
        return self.info[idx]["observation"]

    def get_action_hint(self, idx: int):
        self._check_id(idx)
        return {
            "possible_actions": self.env[idx].get_possible_actions(),
            "possible_objects": self.env[idx].get_possible_objects(),
        }

    def get_goals(self, idx: int):
        self._check_id(idx)
        return {"goals": self.env[idx].get_goal_progress_str()}

    def get_detailed_info(self, idx: int):
        self._check_id(idx)
        return self.info[idx]

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise EnvNotFoundError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise EnvClosedError(
                f"The task with environment {idx} has been deleted."
            )
        if not is_reset and self.info[idx]["done"]:
            raise EpisodeFinishedError(
                f"The task with environment {idx} has finished."
            )

    def close(self, idx: int):
        if idx not in self.info:
            raise EnvNotFoundError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise EnvClosedError(
                f"The task with environment {idx} has been deleted."
            )
        self.env[idx].close()
        self.info[idx]["deleted"] = True
        self.ls.remove(idx)
        print(f"-------Env {idx} closed--------")
        return True

    def get_task_description(self, idx: int):
        self._check_id(idx)
        task_desc = self.env[idx].get_task_description()
        return {"task_description": task_desc}

    def get_object_tree(self, idx: int):
        self._check_id(idx)
        object_tree = self.env[idx].get_object_tree()
        return {"object_tree": object_tree}

    def get_current_state(self, idx: int):
        self._check_id(idx)
        state = {
            "observation": self.env[idx].look(),
            "inventory": self.env[idx].inventory(),
            "task_description": self.env[idx].get_task_description(),
            "goal_progress": self.env[idx].get_goal_progress(),
            "possible_actions": self.env[idx].get_possible_actions()[:10],
            "possible_objects": self.env[idx].get_possible_objects()[:10],
            "current_moves": self.env[idx].get_num_moves(),
            "environment_info": self.info[idx],
        }
        return state
