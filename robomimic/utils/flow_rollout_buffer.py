"""
Minimal rollout buffer for segment-level reward-weighted flow finetuning.
"""
from copy import deepcopy

import numpy as np


class FlowRolloutBuffer:
    def __init__(self, action_horizon, topk_fraction=1.0, use_segment_level_weighting=True):
        self.action_horizon = action_horizon
        self.topk_fraction = topk_fraction
        self.use_segment_level_weighting = use_segment_level_weighting
        self.episodes = []
        self.segments = []

    def add_episode(self, episode):
        self.episodes.append(deepcopy(episode))

    def clear(self):
        self.episodes = []
        self.segments = []

    def compute_returns(self, gamma):
        self.segments = []
        for episode in self.episodes:
            step_rewards = [float(step["reward"]) for step in episode["steps"]]
            step_returns = np.zeros(len(step_rewards), dtype=np.float32)
            running = 0.0
            for idx in reversed(range(len(step_rewards))):
                running = step_rewards[idx] + gamma * running
                step_returns[idx] = running
            episode_return = float(step_returns[0]) if len(step_returns) > 0 else 0.0
            for segment in episode["segments"]:
                seg = deepcopy(segment)
                if len(step_returns) == 0:
                    seg_return = 0.0
                elif self.use_segment_level_weighting:
                    seg_return = float(step_returns[seg["start_step"]])
                else:
                    seg_return = episode_return
                seg["return"] = seg_return
                seg["episode_return"] = episode_return
                self.segments.append(seg)
        return self.segments

    def normalize_advantages(self, eps=1e-6):
        if len(self.segments) == 0:
            return
        returns = np.array([seg["return"] for seg in self.segments], dtype=np.float32)
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        denom = std + eps
        for seg in self.segments:
            seg["advantage"] = float((seg["return"] - mean) / denom)

    def compute_weights(self, temperature, min_weight, max_weight, topk_fraction=None):
        if len(self.segments) == 0:
            return
        topk_fraction = self.topk_fraction if topk_fraction is None else topk_fraction
        advantages = np.array([seg.get("advantage", 0.0) for seg in self.segments], dtype=np.float32)
        weights = np.exp(advantages / max(float(temperature), 1e-6))
        weights = np.clip(weights, min_weight, max_weight)

        if topk_fraction is not None and 0.0 < topk_fraction < 1.0:
            num_keep = max(1, int(np.ceil(topk_fraction * len(weights))))
            keep_indices = np.argsort(weights)[-num_keep:]
            keep_mask = np.zeros(len(weights), dtype=bool)
            keep_mask[keep_indices] = True
            weights = np.where(keep_mask, weights, 0.0)

        for seg, weight in zip(self.segments, weights):
            seg["weight"] = float(weight)

    def sample_batch(self, batch_size, only_positive_weights=True):
        assert len(self.segments) > 0, "rollout buffer is empty"
        if only_positive_weights:
            candidate_indices = [i for i, seg in enumerate(self.segments) if seg.get("weight", 0.0) > 0.0]
            if len(candidate_indices) == 0:
                candidate_indices = list(range(len(self.segments)))
        else:
            candidate_indices = list(range(len(self.segments)))

        replace = len(candidate_indices) < batch_size
        indices = np.random.choice(candidate_indices, size=batch_size, replace=replace)
        selected = [self.segments[i] for i in indices]

        batch = {
            "obs": {},
            "actions": np.stack([seg["action_chunk"] for seg in selected], axis=0),
            "action_mask": np.stack([seg["action_mask"] for seg in selected], axis=0),
            "weights": np.array([seg.get("weight", 1.0) for seg in selected], dtype=np.float32),
            "returns": np.array([seg.get("return", 0.0) for seg in selected], dtype=np.float32),
            "advantages": np.array([seg.get("advantage", 0.0) for seg in selected], dtype=np.float32),
            "reward": np.array([seg.get("segment_reward", 0.0) for seg in selected], dtype=np.float32),
            "done": np.array([seg.get("done", False) for seg in selected], dtype=np.float32),
            "success": np.array([seg.get("success", False) for seg in selected], dtype=np.float32),
            "episode_id": np.array([seg["episode_id"] for seg in selected], dtype=np.int64),
            "timestep": np.array([seg["timestep"] for seg in selected], dtype=np.int64),
            "goal_obs": None,
        }
        obs_keys = selected[0]["obs"].keys()
        for key in obs_keys:
            batch["obs"][key] = np.stack([seg["obs"][key] for seg in selected], axis=0)

        staged_keys = set()
        for seg in selected:
            staged_keys.update(seg.get("staged_summary", {}).keys())
        if len(staged_keys) > 0:
            batch["staged_rewards"] = {
                key: np.array([seg.get("staged_summary", {}).get(key, 0.0) for seg in selected], dtype=np.float32)
                for key in sorted(staged_keys)
            }
        else:
            batch["staged_rewards"] = {}
        return batch

    def get_stats(self):
        num_episodes = len(self.episodes)
        num_segments = len(self.segments)
        if num_segments == 0:
            return {
                "num_episodes": num_episodes,
                "num_segments": 0,
                "success_rate": 0.0,
                "mean_return": 0.0,
                "mean_weight": 0.0,
            }
        success_rate = float(np.mean([float(ep.get("success", False)) for ep in self.episodes])) if num_episodes > 0 else 0.0
        mean_return = float(np.mean([seg.get("return", 0.0) for seg in self.segments]))
        mean_weight = float(np.mean([seg.get("weight", 0.0) for seg in self.segments]))
        return {
            "num_episodes": num_episodes,
            "num_segments": num_segments,
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_weight": mean_weight,
        }

