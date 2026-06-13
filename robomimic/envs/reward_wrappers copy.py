"""
Square NutAssembly stage-transition reward wrapper.

The wrapper converts robosuite's continuous state-shaping values into explicit,
one-time task milestones:

    0: no progress
    1: stable_grasp
    2: lift
    3: hover / peg_align
    4: success

For Square, "hover" is deliberately made geometric and interpretable:
the square nut must remain grasped and lifted, and its center must stay within
``hover_xy_threshold`` metres of the correct peg in the XY plane for
``stable_hover_steps`` consecutive control steps.

The robosuite r_hover score is still recorded for diagnostics and is used only
as a fallback if direct simulator geometry is unavailable.
"""
from collections import OrderedDict

import numpy as np

import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import EnvWrapper


class StagedRewardWrapper(EnvWrapper):
    STAGE_NAMES = {
        0: "none",
        1: "stable_grasp",
        2: "lift",
        3: "hover",
        4: "success",
    }

    def __init__(
        self,
        env,
        use_staged_reward=True,
        success_bonus=5.0,
        grasp_bonus=1.0,
        lift_bonus=1.0,
        hover_bonus=1.0,
        stable_grasp_steps=3,
        stable_hover_steps=3,
        grasp_threshold=0.30,
        lift_threshold=0.40,
        hover_threshold=0.60,
        use_geometric_hover=True,
        hover_xy_threshold=0.05,
    ):
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        super().__init__(env=env)

        if int(stable_grasp_steps) < 1:
            raise ValueError("stable_grasp_steps must be at least 1")
        if int(stable_hover_steps) < 1:
            raise ValueError("stable_hover_steps must be at least 1")
        if min(
            float(grasp_threshold),
            float(lift_threshold),
            float(hover_threshold),
            float(hover_xy_threshold),
        ) < 0.0:
            raise ValueError("stage thresholds must be non-negative")

        self.use_staged_reward = bool(use_staged_reward)

        self.success_bonus = float(success_bonus)
        self.grasp_bonus = float(grasp_bonus)
        self.lift_bonus = float(lift_bonus)
        self.hover_bonus = float(hover_bonus)

        self.stable_grasp_steps = int(stable_grasp_steps)
        self.stable_hover_steps = int(stable_hover_steps)
        self.grasp_threshold = float(grasp_threshold)
        self.lift_threshold = float(lift_threshold)
        self.hover_threshold = float(hover_threshold)

        self.use_geometric_hover = bool(use_geometric_hover)
        self.hover_xy_threshold = float(hover_xy_threshold)

        self._reset_episode_state()

    def _reset_episode_state(self):
        self._prev_success = False
        self._grasp_streak = 0
        self._hover_streak = 0
        self._max_stage_id = 0
        self._episode_step = 0

    def reset(self):
        self._reset_episode_state()
        return self.env.reset()

    def reset_to(self, state):
        self._reset_episode_state()
        return self.env.reset_to(state)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        info = dict(info)

        success = self._safe_success(info)
        staged = self._safe_staged_rewards()
        geometry = self._safe_square_geometry()

        current_stage_id, stage_flags = self._infer_current_stage(
            staged=staged,
            success=success,
            geometry=geometry,
        )

        previous_max_stage_id = int(self._max_stage_id)
        new_max_stage_id = max(previous_max_stage_id, int(current_stage_id))
        newly_achieved_stage_ids = list(
            range(previous_max_stage_id + 1, new_max_stage_id + 1)
        )

        if self.use_staged_reward and len(staged) > 0:
            stage_rewards = self._compute_transition_rewards(
                newly_achieved_stage_ids
            )
            progress_reward = (
                stage_rewards["grasp"]
                + stage_rewards["lift"]
                + stage_rewards["hover"]
            )
            success_reward = stage_rewards["success"]
            reward_total = progress_reward + success_reward
        else:
            progress_reward = float(env_reward)
            success_reward = (
                self.success_bonus
                if success and not self._prev_success
                else 0.0
            )
            reward_total = progress_reward + success_reward
            stage_rewards = {
                "grasp": 0.0,
                "lift": 0.0,
                "hover": 0.0,
                "success": float(success_reward),
            }

        self._max_stage_id = new_max_stage_id

        info.update(
            self._format_reward_info(
                env_reward=env_reward,
                staged=staged,
                success=success,
                stage_flags=stage_flags,
                geometry=geometry,
                current_stage_id=current_stage_id,
                previous_max_stage_id=previous_max_stage_id,
                max_stage_id=new_max_stage_id,
                newly_achieved_stage_ids=newly_achieved_stage_ids,
                stage_rewards=stage_rewards,
                progress_reward=progress_reward,
                success_reward=success_reward,
                reward_total=reward_total,
                episode_step=self._episode_step,
            )
        )

        self._prev_success = bool(success)
        self._episode_step += 1
        return obs, float(reward_total), done, info

    def _base_robosuite_env(self):
        return getattr(self.unwrapped, "base_env", None)

    def _safe_success(self, info):
        for key in ("success", "is_success"):
            if key in info:
                success = info[key]
                if isinstance(success, dict):
                    return bool(success.get("task", False))
                return bool(success)
        try:
            success = self.env.is_success()
            if isinstance(success, dict):
                return bool(success.get("task", False))
            return bool(success)
        except Exception:
            return False

    def _safe_staged_rewards(self):
        if not self.use_staged_reward:
            return OrderedDict()

        base_env = self._base_robosuite_env()
        if base_env is None or not hasattr(base_env, "staged_rewards"):
            return OrderedDict()

        try:
            staged = base_env.staged_rewards()
        except Exception:
            return OrderedDict()

        if isinstance(staged, dict):
            return OrderedDict(
                (str(key), float(value))
                for key, value in staged.items()
            )

        if isinstance(staged, (list, tuple)):
            keys = ("r_reach", "r_grasp", "r_lift", "r_hover")
            return OrderedDict(
                (
                    keys[index] if index < len(keys) else f"r_stage_{index}",
                    float(value),
                )
                for index, value in enumerate(staged)
            )

        return OrderedDict()

    def _safe_square_geometry(self):
        """
        Read SquareNut and square-peg positions directly from robosuite.

        Returns finite scalar diagnostics when available. Failure is non-fatal;
        the wrapper then falls back to the original r_hover threshold.
        """
        result = {
            "geometry_available": False,
            "hover_xy_dist": float("nan"),
            "nut_height_above_table": float("nan"),
            "nut_x": float("nan"),
            "nut_y": float("nan"),
            "nut_z": float("nan"),
            "peg_x": float("nan"),
            "peg_y": float("nan"),
            "peg_z": float("nan"),
        }
        base_env = self._base_robosuite_env()
        if base_env is None:
            return result

        try:
            nut_name = getattr(base_env, "obj_to_use", None)
            if nut_name is None:
                candidate_names = list(getattr(base_env, "obj_body_id", {}).keys())
                square_names = [
                    name for name in candidate_names
                    if "square" in str(name).lower()
                ]
                if not square_names:
                    return result
                nut_name = square_names[0]

            nut_body_id = base_env.obj_body_id[nut_name]
            # In NutAssembly, square maps to nut id 0 and peg1.
            peg_body_id = base_env.peg1_body_id

            nut_pos = np.asarray(
                base_env.sim.data.body_xpos[nut_body_id],
                dtype=np.float64,
            ).copy()
            peg_pos = np.asarray(
                base_env.sim.data.body_xpos[peg_body_id],
                dtype=np.float64,
            ).copy()

            table_offset = np.asarray(
                getattr(base_env, "table_offset", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            xy_dist = float(np.linalg.norm(nut_pos[:2] - peg_pos[:2]))

            result.update({
                "geometry_available": True,
                "hover_xy_dist": xy_dist,
                "nut_height_above_table": float(
                    nut_pos[2] - table_offset[2]
                ),
                "nut_x": float(nut_pos[0]),
                "nut_y": float(nut_pos[1]),
                "nut_z": float(nut_pos[2]),
                "peg_x": float(peg_pos[0]),
                "peg_y": float(peg_pos[1]),
                "peg_z": float(peg_pos[2]),
            })
        except Exception:
            pass

        return result

    def _infer_current_stage(self, staged, success, geometry):
        r_grasp = float(staged.get("r_grasp", 0.0))
        r_lift = float(staged.get("r_lift", 0.0))
        r_hover = float(staged.get("r_hover", 0.0))

        grasp_contact = r_grasp >= self.grasp_threshold
        if grasp_contact:
            self._grasp_streak += 1
        else:
            self._grasp_streak = 0
        stable_grasp = self._grasp_streak >= self.stable_grasp_steps

        lifted = bool(
            stable_grasp and r_lift >= self.lift_threshold
        )

        geometry_available = bool(
            geometry.get("geometry_available", False)
        )
        hover_xy_dist = float(
            geometry.get("hover_xy_dist", float("nan"))
        )

        if (
            self.use_geometric_hover
            and geometry_available
            and np.isfinite(hover_xy_dist)
        ):
            peg_aligned_now = bool(
                lifted and hover_xy_dist <= self.hover_xy_threshold
            )
            hover_detection_source = "geometry"
        else:
            peg_aligned_now = bool(
                lifted and r_hover >= self.hover_threshold
            )
            hover_detection_source = "r_hover_fallback"

        if peg_aligned_now:
            self._hover_streak += 1
        else:
            self._hover_streak = 0

        hovering = bool(
            self._hover_streak >= self.stable_hover_steps
        )

        if success:
            current_stage_id = 4
        elif hovering:
            current_stage_id = 3
        elif lifted:
            current_stage_id = 2
        elif stable_grasp:
            current_stage_id = 1
        else:
            current_stage_id = 0

        stage_flags = {
            "grasp_contact": bool(grasp_contact),
            "stable_grasp": bool(stable_grasp),
            "lifted": bool(lifted),
            "peg_aligned_now": bool(peg_aligned_now),
            "hovering": bool(hovering),
            "hover_detection_source": hover_detection_source,
        }
        return int(current_stage_id), stage_flags

    def _compute_transition_rewards(self, newly_achieved_stage_ids):
        rewards = {
            "grasp": 0.0,
            "lift": 0.0,
            "hover": 0.0,
            "success": 0.0,
        }
        for stage_id in newly_achieved_stage_ids:
            if stage_id == 1:
                rewards["grasp"] += self.grasp_bonus
            elif stage_id == 2:
                rewards["lift"] += self.lift_bonus
            elif stage_id == 3:
                rewards["hover"] += self.hover_bonus
            elif stage_id == 4:
                rewards["success"] += self.success_bonus
        return rewards

    def _format_reward_info(
        self,
        env_reward,
        staged,
        success,
        stage_flags,
        geometry,
        current_stage_id,
        previous_max_stage_id,
        max_stage_id,
        newly_achieved_stage_ids,
        stage_rewards,
        progress_reward,
        success_reward,
        reward_total,
        episode_step,
    ):
        new_stage = len(newly_achieved_stage_ids) > 0
        new_stage_id = (
            newly_achieved_stage_ids[-1]
            if new_stage
            else previous_max_stage_id
        )

        info = {
            "base_reward": float(env_reward),
            "r_reach": 0.0,
            "r_grasp": 0.0,
            "r_lift": 0.0,
            "r_hover": 0.0,
            "success": bool(success),

            "r_grasp_transition": float(stage_rewards["grasp"]),
            "r_lift_transition": float(stage_rewards["lift"]),
            "r_hover_transition": float(stage_rewards["hover"]),
            "r_success": float(success_reward),
            "reward_progress": float(progress_reward),
            "stage_transition_reward": float(reward_total),
            "reward_total": float(reward_total),

            "episode_step": int(episode_step),
            "current_stage_id": int(current_stage_id),
            "current_stage_name": self.STAGE_NAMES[current_stage_id],
            "max_stage_id": int(max_stage_id),
            "max_stage_name": self.STAGE_NAMES[max_stage_id],
            "new_stage": bool(new_stage),
            "new_stage_id": int(new_stage_id),
            "new_stage_name": self.STAGE_NAMES[new_stage_id],
            "newly_achieved_stage_ids": tuple(
                int(stage_id) for stage_id in newly_achieved_stage_ids
            ),

            "grasp_streak": int(self._grasp_streak),
            "hover_streak": int(self._hover_streak),
            "stable_grasp_steps_required": int(
                self.stable_grasp_steps
            ),
            "stable_hover_steps_required": int(
                self.stable_hover_steps
            ),
            "hover_xy_threshold": float(self.hover_xy_threshold),
            **stage_flags,
        }

        for key, value in geometry.items():
            if isinstance(value, (bool, np.bool_)):
                info[key] = bool(value)
            elif isinstance(value, str):
                info[key] = value
            else:
                info[key] = float(value)

        for key, value in staged.items():
            info[key] = float(value)

        return info
