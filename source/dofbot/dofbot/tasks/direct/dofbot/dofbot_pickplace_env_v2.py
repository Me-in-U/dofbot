# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-Place V2: Curriculum Learning Implementation
- Stage-based reward shaping
- Contact-based grasp detection
- Better exploration and learning
"""

from __future__ import annotations

import importlib
import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils  # type: ignore[import]
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # type: ignore[import]
from isaaclab.envs import DirectRLEnv  # type: ignore[import]
from pxr import UsdGeom  # type: ignore[import]

if TYPE_CHECKING:
    from isaacsim.core.utils import prims as prim_utils  # type: ignore[import]
else:
    prim_utils = importlib.import_module("isaacsim.core.utils.prims")

from .dofbot_pickplace_env_cfg_v2 import DofbotPickPlaceEnvCfgV2


class DofbotPickPlaceEnvV2(DirectRLEnv):
    """
    Dofbot pick-and-place V2 환경 구현

    주요 개선:
    1. Curriculum learning with 5 stages
    2. Contact-based grasp detection
    3. Simplified, stage-based rewards
    4. Better observation space (includes object velocity)

    Stages:
    - Stage 1: Reach (EE → Object)
    - Stage 2: Grasp (close gripper around object)
    - Stage 3: Lift (raise object off table)
    - Stage 4: Transport (move grasped object → goal)
    - Stage 5: Place (release at goal)

    관측:
        - q(5), dq(5)
        - ee_pos(3), obj_pos(3), goal_pos(3)
        - grip_joint(1), obj_vel(3)

    액션:
        - 6차원: 5개 관절 속도 + 1개 그리퍼 속도
    """

    cfg: DofbotPickPlaceEnvCfgV2

    def __init__(
        self, cfg: DofbotPickPlaceEnvCfgV2, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve indices
        self._arm_dof_idx, _ = self.robot.find_joints(
            self.cfg.arm_dof_names, preserve_order=True
        )
        self._grip_dof_idx, _ = self.robot.find_joints(
            self.cfg.gripper_dof_name, preserve_order=True
        )
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.ee_link_name)

        # Shortcuts
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Caches
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # Grasp state tracking
        self._grasp_state = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Stage progression tracking (for curriculum learning)
        self._current_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Stages: 0=reach, 1=grasp, 2=lift, 3=transport, 4=place

        # Previous action for jerk penalty
        self._prev_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )

        # Placement constraints
        axis = getattr(self.cfg, "placement_front_axis", "x")
        if isinstance(axis, str):
            axis = axis.lower()
        else:
            axis = "x"
        if axis not in {"x", "y"}:
            axis = "x"
        self._front_axis = 0 if axis == "x" else 1
        self._front_only = bool(getattr(self.cfg, "placement_front_only", False))
        self._object_min_radius = float(getattr(self.cfg, "object_min_radius", 0.0))
        self._goal_min_radius = float(getattr(self.cfg, "goal_min_radius", 0.0))
        self._goal_object_min_sep = float(
            getattr(self.cfg, "goal_object_min_separation", 0.0)
        )

        self._configure_joint_drives()

    # Scene ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal = RigidObject(self.cfg.goal_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self._hide_env_ground_planes()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal"] = self.goal

        # Brighter light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _hide_env_ground_planes(self) -> None:
        stage = self.sim.stage
        patterns = [
            "/World/envs/env_.*/GroundPlane",
            "/World/envs/env_.*/defaultGroundPlane",
            "/World/envs/env_.*/Robot/GroundPlane",
            "/World/envs/env_.*/Robot/defaultGroundPlane",
        ]
        hidden = set()
        for pattern in patterns:
            matches = sim_utils.find_matching_prim_paths(pattern, stage=stage)
            for matched_path in matches:
                if matched_path in hidden:
                    continue
                prim = stage.GetPrimAtPath(matched_path)
                if not prim or not prim.IsValid():
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                hidden.add(matched_path)

    def _configure_joint_drives(self):
        from pxr import UsdPhysics  # type: ignore

        stage = self.sim.stage
        for env_idx in range(self.num_envs):
            for joint_name in self.cfg.arm_dof_names + [self.cfg.gripper_dof_name]:
                joint_path = f"/World/envs/env_{env_idx}/Robot/yahboom_dofbot/joints/{joint_name}"
                joint_prim = stage.GetPrimAtPath(joint_path)
                if joint_prim and joint_prim.IsValid():
                    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

    # Control/Step -----------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        arm_velocities = self.actions[:, :5] * self.cfg.joint_velocity_scale
        grip_velocity = self.actions[:, 5:6] * self.cfg.grip_velocity_scale

        all_velocities = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        all_velocities[:, self._arm_dof_idx] = arm_velocities
        all_velocities[:, self._grip_dof_idx] = grip_velocity

        self.robot.set_joint_velocity_target(all_velocities)

    # Observations/Rewards/Dones --------------------------------------------
    def _get_observations(self) -> dict:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]].unsqueeze(-1)

        self._update_asset_states()
        obj_pos_w = self.obj_pos_w
        goal_pos_w = self.goal_pos_w

        # Object velocity (world frame)
        obj_vel_w = self.object.data.root_lin_vel_w

        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_dof_idx].reshape(self.num_envs, -1),  # 5
                self.joint_vel[:, self._arm_dof_idx].reshape(self.num_envs, -1),  # 5
                ee_pos_w,  # 3
                obj_pos_w,  # 3
                goal_pos_w,  # 3
                grip,  # 1
                obj_vel_w,  # 3
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Curriculum-based reward with 5 stages:
        1. Reach: guide EE to object
        2. Grasp: close gripper around object
        3. Lift: raise object off table
        4. Transport: move object to goal
        5. Place: release at goal
        """
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]]

        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        # Distances
        d_ee_obj = torch.linalg.norm(ee_pos_w - obj_pos_w, dim=-1)
        d_obj_goal = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)

        # Object height above table
        table_contact = self.cfg.table_height + 0.5 * self.cfg.object_size
        obj_height = torch.clamp(obj_pos_w[:, 2] - table_contact, min=0.0)

        # Grasp conditions
        gripper_closed = grip < self.cfg.grip_closed_threshold
        gripper_open = grip > self.cfg.grip_open_threshold
        near_object = d_ee_obj < self.cfg.reach_threshold

        # Update grasp state
        grasp_candidate = (d_ee_obj < self.cfg.grasp_threshold) & gripper_closed
        release_condition = gripper_open | ((obj_height < 0.005) & (~grasp_candidate))
        self._grasp_state |= grasp_candidate
        self._grasp_state &= ~release_condition

        # Stage progression
        reached = d_ee_obj < self.cfg.reach_threshold
        grasped = self._grasp_state
        lifted = obj_height > self.cfg.lift_threshold
        near_goal = d_obj_goal < self.cfg.place_threshold

        # Update stage tracking
        self._current_stage = torch.where(reached, 1, 0)
        self._current_stage = torch.where(grasped, 2, self._current_stage)
        self._current_stage = torch.where(lifted & grasped, 3, self._current_stage)
        self._current_stage = torch.where(near_goal & grasped, 4, self._current_stage)

        # Initialize reward
        reward = torch.full((self.num_envs,), self.cfg.rew_alive, device=self.device)

        # ===== Stage 1: Reach =====
        reward += self.cfg.rew_stage1_reach * d_ee_obj
        reward += self.cfg.rew_stage1_bonus * reached.float()

        # ===== Stage 2: Grasp =====
        if near_object.any():
            # Encourage closing gripper when near object
            gripper_closure = torch.clamp(
                (self.cfg.grip_closed_threshold - grip)
                / abs(self.cfg.grip_closed_threshold - self.cfg.grip_open_threshold),
                min=0.0,
                max=1.0,
            )
            reward += (
                self.cfg.rew_stage2_close_gripper
                * near_object.float()
                * gripper_closure
            )

            # Penalize keeping gripper open near object
            reward += (
                self.cfg.rew_stage2_hold_penalty
                * near_object.float()
                * gripper_open.float()
            )

        # Bonus for successful grasp
        reward += self.cfg.rew_stage2_grasp_bonus * grasped.float()

        # ===== Stage 3: Lift =====
        if grasped.any():
            # Reward lifting the object
            lift_progress = torch.clamp(obj_height / self.cfg.lift_threshold, max=1.0)
            reward += self.cfg.rew_stage3_lift * grasped.float() * lift_progress

            # Bonus for reaching lift height
            reward += self.cfg.rew_stage3_bonus * (grasped & lifted).float()

        # ===== Stage 4: Transport =====
        if (grasped & lifted).any():
            # Reward moving object toward goal
            reward += (
                self.cfg.rew_stage4_transport * (grasped & lifted).float() * d_obj_goal
            )

            # Progress reward (closer to goal = higher reward)
            goal_proximity = torch.exp(-2.0 * d_obj_goal)
            reward += (
                self.cfg.rew_stage4_progress
                * (grasped & lifted).float()
                * goal_proximity
            )

        # ===== Stage 5: Place =====
        # Final success bonus
        reward += self.cfg.rew_stage5_place_bonus * (grasped & near_goal).float()

        # ===== General penalties =====
        # Action magnitude penalty (encourage efficiency)
        action_magnitude = torch.sum(torch.abs(self.actions), dim=-1)
        reward += self.cfg.rew_penalty_action * action_magnitude

        # Action smoothness penalty (reduce jerk)
        action_diff = torch.sum(torch.abs(self.actions - self._prev_actions), dim=-1)
        reward += self.cfg.rew_penalty_jerk * action_diff
        self._prev_actions = self.actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success: object at goal while grasped
        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        d_place = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)
        success = (d_place < self.cfg.success_threshold) & self._grasp_state

        return success, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_tensor = self._to_env_tensor(env_ids)
        env_count = env_tensor.shape[0]

        # Reset grasp and stage tracking
        self._grasp_state[env_tensor] = False
        self._current_stage[env_tensor] = 0
        self._prev_actions[env_tensor] = 0.0

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        noise_pos = self.cfg.joint_pos_noise * (2.0 * torch.rand_like(joint_pos) - 1.0)
        noise_vel = self.cfg.joint_vel_noise * (2.0 * torch.rand_like(joint_vel) - 1.0)
        joint_pos = joint_pos + noise_pos
        joint_vel = joint_vel + noise_vel

        default_root = self.robot.data.default_root_state[env_ids]
        default_root[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset object & goal
        obj_xy = self._sample_xy(
            env_ids,
            self.cfg.object_xy_range,
            min_radius=self._object_min_radius,
            front_only=self._front_only,
        )
        obj_z = torch.full(
            (env_count, 1),
            self.cfg.table_height + 0.5 * self.cfg.object_size,
            device=self.device,
        )
        obj_pos_local = torch.cat((obj_xy, obj_z), dim=-1)

        goal_xy = self._sample_goal_xy(env_ids, obj_xy)
        goal_z = torch.full((env_count, 1), self.cfg.goal_height, device=self.device)
        goal_pos_local = torch.cat((goal_xy, goal_z), dim=-1)

        obj_pos_w = obj_pos_local + self.scene.env_origins[env_ids]
        goal_pos_w = goal_pos_local + self.scene.env_origins[env_ids]

        object_state = torch.zeros((env_count, 13), device=self.device)
        object_state[:, :3] = obj_pos_w
        object_state[:, 3] = 1.0

        goal_state = torch.zeros((env_count, 13), device=self.device)
        goal_state[:, :3] = goal_pos_w
        goal_state[:, 3] = 1.0

        self.obj_pos_w[env_tensor] = obj_pos_local
        self.goal_pos_w[env_tensor] = goal_pos_local

        self.object.write_root_state_to_sim(object_state, env_ids)
        self.goal.write_root_state_to_sim(goal_state, env_ids)

    # Utils ---------------------------------------------------------------
    def _sample_xy(
        self,
        env_ids: Sequence[int],
        xy_range: tuple[float, float],
        min_radius: float = 0.0,
        front_only: bool = False,
    ) -> torch.Tensor:
        low, high = xy_range
        max_radius = max(abs(low), abs(high))
        max_radius = max(max_radius, 1e-4)
        radial_min = max(0.0, float(min_radius))
        if radial_min >= max_radius:
            radial_min = max_radius - 1e-5
        span = max(max_radius - radial_min, 1e-5)

        env_tensor = self._to_env_tensor(env_ids)
        front_flag = front_only or getattr(self, "_front_only", False)

        base = torch.rand((env_tensor.shape[0], 3), device=self.device)
        hash_offsets = torch.frac(
            torch.stack(
                (
                    env_tensor.float() * 0.61803398875,
                    env_tensor.float() * 0.38196601125,
                    env_tensor.float() * 0.17320508076,
                ),
                dim=-1,
            )
        )
        samples = torch.frac(base + hash_offsets)

        radius = radial_min + span * samples[:, 0]
        if front_flag:
            angles = (samples[:, 1] - 0.5) * math.pi
        else:
            angles = samples[:, 1] * 2.0 * math.pi

        forward = radius * torch.cos(angles)
        lateral = radius * torch.sin(angles)

        if front_flag:
            forward = torch.clamp_min(forward, 1e-4)

        coords = torch.stack((forward, lateral), dim=-1)
        if getattr(self, "_front_axis", 0) == 1:
            coords = coords[:, [1, 0]]

        coords = coords.clamp(min=low, max=high)
        return coords

    def _sample_goal_xy(
        self, env_ids: Sequence[int], obj_xy: torch.Tensor
    ) -> torch.Tensor:
        env_tensor = self._to_env_tensor(env_ids)
        goal_xy = torch.zeros((env_tensor.shape[0], 2), device=self.device)

        remaining_pos = torch.arange(
            env_tensor.shape[0], device=self.device, dtype=torch.long
        )
        attempts = 0
        max_attempts = 24
        min_sep = max(0.0, self._goal_object_min_sep)
        separation_tol = min_sep + 0.01 if min_sep > 0.0 else 0.0

        while remaining_pos.numel() > 0 and attempts < max_attempts:
            env_subset = env_tensor[remaining_pos]
            candidates = self._sample_xy(
                env_subset,
                self.cfg.goal_xy_range,
                min_radius=self._goal_min_radius,
                front_only=self._front_only,
            )
            if min_sep > 0.0:
                dists = torch.linalg.norm(candidates - obj_xy[remaining_pos], dim=-1)
                mask = dists >= separation_tol
            else:
                mask = torch.ones(
                    candidates.shape[0], dtype=torch.bool, device=self.device
                )

            if mask.any():
                valid_pos = remaining_pos[mask]
                goal_xy[valid_pos] = candidates[mask]
                remaining_pos = remaining_pos[~mask]
            attempts += 1

        # Fallback
        if remaining_pos.numel() > 0:
            obj_rem = obj_xy[remaining_pos]
            axis = self._front_axis
            lat_axis = 1 - axis
            low, high = self.cfg.goal_xy_range
            front_min = max(min_sep, 1e-4)

            fallback = obj_rem.clone()
            forward_offset = separation_tol + 0.05
            fallback[:, axis] = torch.clamp(
                obj_rem[:, axis] + forward_offset, min=front_min, max=high
            )

            rand_sign = torch.where(
                torch.rand_like(obj_rem[:, lat_axis]) > 0.5, 1.0, -1.0
            )
            lateral_offset = separation_tol + 0.05
            fallback[:, lat_axis] = obj_rem[:, lat_axis] + rand_sign * lateral_offset
            fallback[:, lat_axis] = torch.clamp(
                fallback[:, lat_axis], min=low, max=high
            )

            diff = fallback - obj_rem
            dist = torch.linalg.norm(diff, dim=-1)
            short_mask = dist < min_sep
            if short_mask.any():
                scale_factor = (min_sep + 0.01) / dist[short_mask].clamp(min=1e-6)
                fallback[short_mask] = obj_rem[short_mask] + diff[
                    short_mask
                ] * scale_factor.unsqueeze(-1)
                fallback[:, axis] = torch.clamp(
                    fallback[:, axis], min=front_min, max=high
                )
                fallback[:, lat_axis] = torch.clamp(
                    fallback[:, lat_axis], min=low, max=high
                )

            goal_xy[remaining_pos] = fallback

        return goal_xy

    def _to_env_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            env_tensor = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        elif isinstance(env_ids, torch.Tensor):
            env_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_tensor = torch.tensor(
                list(env_ids), device=self.device, dtype=torch.long
            )
        return env_tensor.reshape(-1)

    def _update_asset_states(self) -> None:
        torch.sub(
            self.object.data.root_pos_w, self.scene.env_origins, out=self.obj_pos_w
        )
        torch.sub(
            self.goal.data.root_pos_w, self.scene.env_origins, out=self.goal_pos_w
        )
