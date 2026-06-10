"""
Reward wrappers for online finetuning.
"""
from collections import OrderedDict

import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import EnvWrapper


class StagedRewardWrapper(EnvWrapper):
    """
    Wrap an environment to expose a uniform staged-reward interface for online
    rollout collection without modifying the underlying robosuite task code.

    Returned step rewards are training rewards:
        staged_progress + success_bonus_on_first_success

    If staged rewards are not available, the wrapper falls back to the
    environment reward and success flag.
    """

    def __init__(self, env, use_staged_reward=True, success_bonus=0.0):
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        super(StagedRewardWrapper, self).__init__(env=env)
        self.use_staged_reward = use_staged_reward
        self.success_bonus = success_bonus
        self._prev_success = False

    def reset(self):
        self._prev_success = False
        return self.env.reset()

    def reset_to(self, state):
        self._prev_success = False
        return self.env.reset_to(state)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        info = dict(info)

        success = self._safe_success(info)
        staged = self._safe_staged_rewards()
        progress_reward = self._compute_progress_reward(env_reward, staged)
        success_reward = self.success_bonus if (success and not self._prev_success) else 0.0
        reward_total = progress_reward + success_reward

        info.update(self._format_reward_info(
            env_reward=env_reward,
            staged=staged,
            success=success,
            success_reward=success_reward,
            reward_total=reward_total,
        ))
        self._prev_success = success
        return obs, reward_total, done, info

    def _safe_success(self, info):
        if "success" in info:
            return bool(info["success"])
        if "is_success" in info:
            return bool(info["is_success"])
        try:
            return bool(self.env.is_success()["task"])
        except Exception:
            return False

    def _safe_staged_rewards(self):
        if not self.use_staged_reward:
            return OrderedDict()

        base_env = getattr(self.unwrapped, "base_env", None)
        if base_env is None or not hasattr(base_env, "staged_rewards"):
            return OrderedDict()

        try:
            staged = base_env.staged_rewards()
        except Exception:
            return OrderedDict()

        if isinstance(staged, dict):
            return OrderedDict((str(k), float(v)) for k, v in staged.items())

        if isinstance(staged, (list, tuple)):
            keys = ("r_reach", "r_grasp", "r_lift", "r_hover")
            out = OrderedDict()
            for idx, value in enumerate(staged):
                key = keys[idx] if idx < len(keys) else f"r_stage_{idx}"
                out[key] = float(value)
            return out

        return OrderedDict()

    def _compute_progress_reward(self, env_reward, staged):
        if not self.use_staged_reward or len(staged) == 0:
            return float(env_reward)
        return float(sum(max(0.0, float(v)) for v in staged.values()))

    @staticmethod
    def _format_reward_info(env_reward, staged, success, success_reward, reward_total):
        info = {
            "base_reward": float(env_reward),
            "r_reach": 0.0,
            "r_grasp": 0.0,
            "r_lift": 0.0,
            "r_hover": 0.0,
            "r_success": float(success_reward),
            "success": bool(success),
            "reward_progress": float(reward_total - success_reward),
            "reward_total": float(reward_total),
        }
        for key, value in staged.items():
            info[key] = float(value)
        return info

