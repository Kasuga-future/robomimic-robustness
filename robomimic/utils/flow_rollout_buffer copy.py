
"""
Stage-interval rollout buffer for curriculum reward-weighted Flow Matching.

Each episode is divided into disjoint action intervals:

    stage 1: episode start          -> first stable grasp
    stage 2: after stable grasp     -> first lift
    stage 3: after lift             -> first hover
    stage 4: after hover            -> first success

Only intervals whose endpoint was actually achieved are trainable. Actions after
the highest achieved stage are discarded. A prediction-horizon source segment
can be copied into multiple stage samples when it crosses a stage boundary; the
masks of those copies are disjoint, so every successful-prefix action is covered
exactly once per online epoch.

The buffer also supports no-replacement mini-batch iteration. Therefore every
positive stage sample is used exactly once per online epoch.
"""
from copy import deepcopy
import math
import numpy as np


class FlowRolloutBuffer:
    STAGE_NAMES = {
        0: "none",
        1: "stable_grasp",
        2: "lift",
        3: "hover",
        4: "success",
    }
    STAGE_KEYS = {
        1: "stable_grasp",
        2: "lift",
        3: "hover",
        4: "success",
    }

    def __init__(
        self,
        action_horizon,
        topk_fraction=1.0,
        use_segment_level_weighting=True,
    ):
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        if topk_fraction is not None:
            topk_fraction = float(topk_fraction)
            if not 0.0 < topk_fraction <= 1.0:
                raise ValueError(
                    "topk_fraction must be in (0, 1] or None, got "
                    f"{topk_fraction}"
                )

        self.topk_fraction = topk_fraction
        self.use_segment_level_weighting = bool(use_segment_level_weighting)
        self.episodes = []
        self.segments = []

        self.num_source_segments = 0
        self.num_eligible_source_segments = 0
        self.num_discarded_source_segments = 0
        self.stage_priorities = {
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
        }

    def add_episode(self, episode):
        self.episodes.append(deepcopy(episode))

    def clear(self):
        self.episodes = []
        self.segments = []
        self.num_source_segments = 0
        self.num_eligible_source_segments = 0
        self.num_discarded_source_segments = 0

    @classmethod
    def _first_success_step(cls, episode):
        for step_idx, step in enumerate(episode.get("steps", [])):
            if bool(step.get("success", False)):
                return int(step_idx)
        return None

    @classmethod
    def _achievement_steps(cls, episode):
        """
        Return a validated mapping stage_id -> first achievement step.
        """
        raw = episode.get("stage_achievement_steps", {}) or {}
        highest_stage_id = int(episode.get("highest_stage_id", 0))
        if bool(episode.get("success", False)):
            highest_stage_id = max(highest_stage_id, 4)
        highest_stage_id = min(max(highest_stage_id, 0), 4)

        result = {}
        previous = -1
        for stage_id in range(1, highest_stage_id + 1):
            key = cls.STAGE_KEYS[stage_id]
            value = raw.get(key, None)

            if value is None and stage_id == 4:
                value = cls._first_success_step(episode)

            if value is None:
                # Compatibility fallback to step-level max_stage_id.
                for step_idx, step in enumerate(episode.get("steps", [])):
                    staged = step.get("staged", {}) or {}
                    if int(staged.get("max_stage_id", 0)) >= stage_id:
                        value = int(step_idx)
                        break

            if value is None:
                raise ValueError(
                    f"episode {episode.get('episode_id')} reached stage "
                    f"{stage_id} ({key}) but has no achievement step"
                )

            value = int(value)
            if value < previous:
                raise ValueError(
                    "stage achievement steps must be non-decreasing, got "
                    f"stage {stage_id} at {value} after {previous}"
                )
            result[stage_id] = value
            previous = value

        return highest_stage_id, result

    def _validate_and_get_step_indices(self, segment):
        action_chunk = np.asarray(segment["action_chunk"])
        action_mask = np.asarray(segment["action_mask"], dtype=np.float32)

        if action_chunk.ndim < 2:
            raise ValueError(
                "action_chunk must have shape [T, action_dim], got "
                f"{action_chunk.shape}"
            )
        if action_chunk.shape[0] != self.action_horizon:
            raise ValueError(
                "segment action length does not match buffer horizon: "
                f"{action_chunk.shape[0]} != {self.action_horizon}"
            )
        if action_mask.shape != (self.action_horizon,):
            raise ValueError(
                f"action_mask must have shape ({self.action_horizon},), "
                f"got {action_mask.shape}"
            )

        if "action_step_indices" in segment:
            step_indices = np.asarray(
                segment["action_step_indices"],
                dtype=np.int64,
            )
        else:
            start_step = int(segment["start_step"])
            execution_start_index = int(
                segment.get("execution_start_index", 0)
            )
            positions = np.arange(self.action_horizon, dtype=np.int64)
            step_indices = start_step + positions - execution_start_index

        if step_indices.shape != (self.action_horizon,):
            raise ValueError(
                "action_step_indices must have shape "
                f"({self.action_horizon},), got {step_indices.shape}"
            )

        return action_mask.copy(), step_indices

    @staticmethod
    def _stage_intervals(highest_stage_id, achievement_steps):
        """
        Build disjoint inclusive intervals [start, end] for achieved stages.
        """
        intervals = []
        start = 0
        for stage_id in range(1, highest_stage_id + 1):
            end = int(achievement_steps[stage_id])
            if end >= start:
                intervals.append((stage_id, start, end))
            # When multiple stages are detected on the same environment step,
            # later intervals can be empty. We keep masks disjoint rather than
            # crediting the same action multiple times.
            start = end + 1
        return intervals

    def compute_returns(self, gamma, stage_priorities=None):
        """
        Split source segments into stage-specific samples and compute a
        discounted closeness score for each sample.

        For target stage s achieved at step e_s, an action at step g receives
        credit gamma ** (e_s - g). The segment return is the mean credit over
        its valid action positions. Cross-stage importance is handled later by
        the curriculum stage priority.
        """
        gamma = float(gamma)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1], got {gamma}")

        if stage_priorities is not None:
            priorities = {}
            for stage_id in range(1, 5):
                value = float(stage_priorities.get(stage_id, 1.0))
                if value <= 0.0:
                    raise ValueError(
                        f"stage priority must be positive, stage {stage_id}: "
                        f"{value}"
                    )
                priorities[stage_id] = value
            self.stage_priorities = priorities

        self.segments = []
        self.num_source_segments = 0
        self.num_eligible_source_segments = 0
        self.num_discarded_source_segments = 0

        for episode in self.episodes:
            highest_stage_id, achievement_steps = self._achievement_steps(
                episode
            )
            intervals = self._stage_intervals(
                highest_stage_id,
                achievement_steps,
            )
            highest_cutoff = (
                achievement_steps[highest_stage_id]
                if highest_stage_id > 0
                else None
            )

            for source_idx, source_segment in enumerate(
                episode.get("segments", [])
            ):
                self.num_source_segments += 1
                original_mask, step_indices = (
                    self._validate_and_get_step_indices(source_segment)
                )
                source_key = (
                    int(episode.get("episode_id", -1)),
                    int(source_segment.get("timestep", source_idx)),
                )

                expected_prefix_mask = np.zeros_like(
                    original_mask,
                    dtype=np.float32,
                )
                if highest_cutoff is not None:
                    expected_prefix_mask = (
                        (original_mask > 0.0)
                        & (step_indices >= 0)
                        & (step_indices <= int(highest_cutoff))
                    ).astype(np.float32)

                union_mask = np.zeros_like(original_mask, dtype=np.float32)
                created = 0

                for stage_id, interval_start, interval_end in intervals:
                    stage_mask = (
                        (original_mask > 0.0)
                        & (step_indices >= int(interval_start))
                        & (step_indices <= int(interval_end))
                    ).astype(np.float32)

                    valid_positions = np.flatnonzero(stage_mask > 0.0)
                    if len(valid_positions) == 0:
                        continue

                    if np.any((union_mask > 0.0) & (stage_mask > 0.0)):
                        raise AssertionError(
                            f"overlapping stage masks for source {source_key}"
                        )
                    union_mask = np.maximum(union_mask, stage_mask)

                    valid_steps = step_indices[valid_positions]
                    distances = int(interval_end) - valid_steps
                    if np.any(distances < 0):
                        raise AssertionError("negative stage distance")

                    per_action_credit = np.power(
                        gamma,
                        distances.astype(np.float32),
                    )
                    if self.use_segment_level_weighting:
                        seg_return = float(np.mean(per_action_credit))
                    else:
                        seg_return = 1.0

                    seg = deepcopy(source_segment)
                    seg["source_segment_id"] = source_key
                    seg["source_segment_index"] = int(source_idx)
                    seg["original_action_mask"] = original_mask.copy()
                    seg["action_mask"] = stage_mask
                    seg["action_step_indices"] = step_indices.copy()

                    seg["eligible"] = True
                    seg["credited_stage_id"] = int(stage_id)
                    seg["credited_stage_name"] = self.STAGE_NAMES[stage_id]
                    seg["stage_interval_start"] = int(interval_start)
                    seg["stage_interval_end"] = int(interval_end)
                    seg["stage_achievement_step"] = int(interval_end)
                    seg["credit_cutoff_step"] = int(interval_end)
                    seg["highest_episode_stage_id"] = int(highest_stage_id)
                    seg["highest_episode_stage_name"] = self.STAGE_NAMES[
                        highest_stage_id
                    ]

                    seg["first_trainable_step"] = int(valid_steps.min())
                    seg["last_trainable_step"] = int(valid_steps.max())
                    seg["num_trainable_actions"] = int(stage_mask.sum())
                    seg["valid_action_fraction"] = float(
                        stage_mask.sum()
                        / max(float(original_mask.sum()), 1.0)
                    )
                    seg["return"] = seg_return
                    seg["episode_return"] = float(
                        sum(
                            self.stage_priorities[s]
                            for s in range(1, highest_stage_id + 1)
                        )
                    )
                    seg["stage_priority"] = float(
                        self.stage_priorities[stage_id]
                    )
                    seg["advantage"] = 0.0
                    seg["weight"] = 0.0

                    self.segments.append(seg)
                    created += 1

                if not np.array_equal(
                    (union_mask > 0.0),
                    (expected_prefix_mask > 0.0),
                ):
                    raise AssertionError(
                        "stage interval masks do not exactly cover successful "
                        f"prefix for source {source_key}: "
                        f"union={union_mask.astype(int).tolist()}, "
                        f"expected={expected_prefix_mask.astype(int).tolist()}"
                    )

                if created > 0:
                    self.num_eligible_source_segments += 1
                else:
                    self.num_discarded_source_segments += 1

        return self.segments

    def normalize_advantages(self, eps=1e-6, group_by_stage=True):
        """
        Normalize returns within each target-stage group.

        This prevents numerous success samples from forcing valid grasp/lift
        samples to have strongly negative global advantages.
        """
        for seg in self.segments:
            seg["advantage"] = 0.0

        if len(self.segments) == 0:
            return

        if group_by_stage:
            groups = {
                stage_id: [
                    idx for idx, seg in enumerate(self.segments)
                    if int(seg["credited_stage_id"]) == stage_id
                ]
                for stage_id in range(1, 5)
            }
        else:
            groups = {
                -1: list(range(len(self.segments)))
            }

        for indices in groups.values():
            if len(indices) == 0:
                continue
            returns = np.asarray(
                [self.segments[idx]["return"] for idx in indices],
                dtype=np.float32,
            )
            mean = float(returns.mean())
            std = float(returns.std())
            if std <= float(eps):
                for idx in indices:
                    self.segments[idx]["advantage"] = 0.0
                continue

            denom = std + float(eps)
            for idx, value in zip(indices, returns):
                self.segments[idx]["advantage"] = float(
                    (float(value) - mean) / denom
                )

    def compute_weights(
        self,
        temperature,
        min_weight,
        max_weight,
        topk_fraction=None,
        stage_priorities=None,
    ):
        """
        Compute curriculum weights:

            weight = stage_priority(stage, iteration)
                     * exp(within_stage_advantage / temperature)

        If top-k is used, it is applied separately inside each stage group.
        For full-data online epochs use topk_fraction=1.0.
        """
        if len(self.segments) == 0:
            return

        temperature = max(float(temperature), 1e-6)
        min_weight = float(min_weight)
        max_weight = float(max_weight)
        if min_weight < 0.0:
            raise ValueError("min_weight must be non-negative")
        if max_weight < min_weight:
            raise ValueError("max_weight must be >= min_weight")

        if stage_priorities is not None:
            self.stage_priorities = {
                stage_id: float(stage_priorities.get(stage_id, 1.0))
                for stage_id in range(1, 5)
            }

        topk_fraction = (
            self.topk_fraction
            if topk_fraction is None
            else float(topk_fraction)
        )
        if topk_fraction is not None and not 0.0 < topk_fraction <= 1.0:
            raise ValueError(
                "topk_fraction must be in (0, 1] or None, got "
                f"{topk_fraction}"
            )

        for seg in self.segments:
            stage_id = int(seg["credited_stage_id"])
            priority = float(self.stage_priorities[stage_id])
            raw_weight = priority * math.exp(
                float(seg.get("advantage", 0.0)) / temperature
            )
            seg["stage_priority"] = priority
            seg["weight"] = float(
                np.clip(raw_weight, min_weight, max_weight)
            )

        if topk_fraction is not None and topk_fraction < 1.0:
            for stage_id in range(1, 5):
                indices = [
                    idx for idx, seg in enumerate(self.segments)
                    if int(seg["credited_stage_id"]) == stage_id
                    and float(seg.get("weight", 0.0)) > 0.0
                ]
                if len(indices) == 0:
                    continue
                num_keep = max(
                    1,
                    int(np.ceil(topk_fraction * len(indices))),
                )
                local_weights = np.asarray(
                    [self.segments[idx]["weight"] for idx in indices],
                    dtype=np.float32,
                )
                keep_local = np.argsort(
                    local_weights,
                    kind="stable",
                )[-num_keep:]
                keep_global = {indices[i] for i in keep_local}
                for idx in indices:
                    if idx not in keep_global:
                        self.segments[idx]["weight"] = 0.0

    def _candidate_indices(self, only_positive_weights=True):
        if only_positive_weights:
            return [
                idx for idx, seg in enumerate(self.segments)
                if float(seg.get("weight", 0.0)) > 0.0
                and float(np.asarray(seg["action_mask"]).sum()) > 0.0
            ]
        return [
            idx for idx, seg in enumerate(self.segments)
            if float(np.asarray(seg["action_mask"]).sum()) > 0.0
        ]

    def has_positive_samples(self):
        return len(self._candidate_indices(True)) > 0

    def _make_batch(self, selected):
        if len(selected) == 0:
            return None

        batch = {
            "obs": {},
            "actions": np.stack(
                [seg["action_chunk"] for seg in selected],
                axis=0,
            ),
            "action_mask": np.stack(
                [seg["action_mask"] for seg in selected],
                axis=0,
            ),
            "weights": np.asarray(
                [seg["weight"] for seg in selected],
                dtype=np.float32,
            ),
            "returns": np.asarray(
                [seg.get("return", 0.0) for seg in selected],
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                [seg.get("advantage", 0.0) for seg in selected],
                dtype=np.float32,
            ),
            "reward": np.asarray(
                [seg.get("segment_reward", 0.0) for seg in selected],
                dtype=np.float32,
            ),
            "done": np.asarray(
                [seg.get("done", False) for seg in selected],
                dtype=np.float32,
            ),
            "success": np.asarray(
                [seg.get("success", False) for seg in selected],
                dtype=np.float32,
            ),
            "episode_id": np.asarray(
                [seg["episode_id"] for seg in selected],
                dtype=np.int64,
            ),
            "timestep": np.asarray(
                [seg["timestep"] for seg in selected],
                dtype=np.int64,
            ),
            "credited_stage_id": np.asarray(
                [seg["credited_stage_id"] for seg in selected],
                dtype=np.int64,
            ),
            "stage_priority": np.asarray(
                [seg["stage_priority"] for seg in selected],
                dtype=np.float32,
            ),
            "goal_obs": None,
        }

        obs_keys = selected[0]["obs"].keys()
        for key in obs_keys:
            batch["obs"][key] = np.stack(
                [seg["obs"][key] for seg in selected],
                axis=0,
            )

        staged_keys = set()
        for seg in selected:
            staged_keys.update(seg.get("staged_summary", {}).keys())
        batch["staged_rewards"] = {
            key: np.asarray(
                [
                    seg.get("staged_summary", {}).get(key, 0.0)
                    for seg in selected
                ],
                dtype=np.float32,
            )
            for key in sorted(staged_keys)
        }
        return batch

    def sample_batch(self, batch_size, only_positive_weights=True):
        """
        Compatibility random sampler. Formal training should prefer
        ``iter_batches`` so every sample is covered without replacement.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            return None
        candidates = self._candidate_indices(only_positive_weights)
        if len(candidates) == 0:
            return None

        replace = len(candidates) < batch_size
        indices = np.random.choice(
            candidates,
            size=batch_size,
            replace=replace,
        )
        return self._make_batch([self.segments[idx] for idx in indices])

    def iter_batches(
        self,
        batch_size,
        num_epochs=1,
        shuffle=True,
        seed=None,
        only_positive_weights=True,
    ):
        """
        Yield mini-batches without replacement within each online epoch.

        Every candidate stage sample is used exactly once per epoch, including
        the final smaller mini-batch.
        """
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)
        if batch_size <= 0 or num_epochs <= 0:
            return

        candidates = np.asarray(
            self._candidate_indices(only_positive_weights),
            dtype=np.int64,
        )
        if len(candidates) == 0:
            return

        rng = np.random.default_rng(seed)
        for online_epoch in range(num_epochs):
            order = candidates.copy()
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                selected = [self.segments[int(idx)] for idx in indices]
                batch = self._make_batch(selected)
                # Do not attach Python scalar metadata to the model batch.
                # robomimic TensorUtils.to_tensor recursively converts every
                # field and does not accept bare Python int values. The trainer
                # tracks progress from the iterator index instead.
                yield batch

    def num_batches(self, batch_size, num_epochs=1):
        num_candidates = len(self._candidate_indices(True))
        if num_candidates == 0 or int(batch_size) <= 0:
            return 0
        return int(math.ceil(num_candidates / int(batch_size))) * int(
            num_epochs
        )

    def get_stats(self):
        num_episodes = len(self.episodes)
        num_stage_segments = len(self.segments)

        episode_stage_ids = np.asarray(
            [
                int(ep.get("highest_stage_id", 0))
                for ep in self.episodes
            ],
            dtype=np.int64,
        )
        positive = [
            seg for seg in self.segments
            if float(seg.get("weight", 0.0)) > 0.0
        ]
        eligible_episode_ids = {
            int(seg["episode_id"]) for seg in self.segments
        }

        success_rate = (
            float(np.mean([
                float(ep.get("success", False))
                for ep in self.episodes
            ]))
            if num_episodes > 0
            else 0.0
        )

        stats = {
            "num_episodes": int(num_episodes),
            "num_source_segments": int(self.num_source_segments),
            "num_eligible_source_segments": int(
                self.num_eligible_source_segments
            ),
            "num_discarded_source_segments": int(
                self.num_discarded_source_segments
            ),
            "num_stage_segments": int(num_stage_segments),
            # Backward-compatible names:
            "num_segments": int(self.num_source_segments),
            "num_eligible_segments": int(num_stage_segments),
            "num_positive_weight_segments": int(len(positive)),
            "num_eligible_episodes": int(len(eligible_episode_ids)),
            "success_rate": success_rate,
            "episode_success_rate": success_rate,
            "num_episodes_no_progress": (
                int(np.sum(episode_stage_ids == 0))
                if num_episodes > 0
                else 0
            ),
            "mean_return": (
                float(np.mean([seg["return"] for seg in self.segments]))
                if num_stage_segments > 0
                else 0.0
            ),
            "mean_weight": (
                float(np.mean([seg["weight"] for seg in positive]))
                if len(positive) > 0
                else 0.0
            ),
            "mean_trainable_actions_per_stage_segment": (
                float(np.mean([
                    seg["num_trainable_actions"] for seg in self.segments
                ]))
                if num_stage_segments > 0
                else 0.0
            ),
        }

        reached_counts = {}
        for stage_id in range(1, 5):
            name = self.STAGE_NAMES[stage_id]
            stage_segments = [
                seg for seg in self.segments
                if int(seg["credited_stage_id"]) == stage_id
            ]
            reached_count = (
                int(np.sum(episode_stage_ids >= stage_id))
                if num_episodes > 0
                else 0
            )
            reached_counts[stage_id] = reached_count

            achievement_values = [
                ep.get("stage_achievement_steps", {}).get(name, None)
                for ep in self.episodes
            ]
            achievement_values = [
                int(value)
                for value in achievement_values
                if value is not None
            ]

            stats[f"num_episodes_reaching_{name}"] = reached_count
            stats[f"rate_reaching_{name}"] = (
                float(reached_count / num_episodes)
                if num_episodes > 0
                else 0.0
            )
            stats[f"num_episodes_ending_at_{name}"] = (
                int(np.sum(episode_stage_ids == stage_id))
                if num_episodes > 0
                else 0
            )
            stats[f"mean_first_step_{name}"] = (
                float(np.mean(achievement_values))
                if achievement_values
                else -1.0
            )
            stats[f"median_first_step_{name}"] = (
                float(np.median(achievement_values))
                if achievement_values
                else -1.0
            )
            stats[f"num_stage_segments_{name}"] = int(
                len(stage_segments)
            )
            stats[f"num_trainable_actions_{name}"] = int(sum(
                int(seg["num_trainable_actions"])
                for seg in stage_segments
            ))
            stats[f"mean_weight_{name}"] = (
                float(np.mean([
                    seg["weight"] for seg in stage_segments
                    if float(seg.get("weight", 0.0)) > 0.0
                ]))
                if any(
                    float(seg.get("weight", 0.0)) > 0.0
                    for seg in stage_segments
                )
                else 0.0
            )
            stats[f"stage_priority_{name}"] = float(
                self.stage_priorities[stage_id]
            )

        stats["lift_given_stable_grasp"] = (
            float(reached_counts[2] / reached_counts[1])
            if reached_counts.get(1, 0) > 0
            else 0.0
        )
        stats["hover_given_lift"] = (
            float(reached_counts[3] / reached_counts[2])
            if reached_counts.get(2, 0) > 0
            else 0.0
        )
        stats["success_given_hover"] = (
            float(reached_counts[4] / reached_counts[3])
            if reached_counts.get(3, 0) > 0
            else 0.0
        )

        hover_distances = []
        for episode in self.episodes:
            hover_step = episode.get(
                "stage_achievement_steps", {}
            ).get("hover", None)
            if hover_step is None:
                continue
            steps = episode.get("steps", [])
            if 0 <= int(hover_step) < len(steps):
                value = steps[int(hover_step)].get(
                    "staged", {}
                ).get("hover_xy_dist", None)
                if value is not None and np.isfinite(float(value)):
                    hover_distances.append(float(value))
        stats["mean_hover_xy_dist_at_first_hover"] = (
            float(np.mean(hover_distances))
            if hover_distances
            else -1.0
        )

        return stats
