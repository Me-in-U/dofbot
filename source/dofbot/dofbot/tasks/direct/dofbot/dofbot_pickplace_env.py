# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils  # type: ignore[import]
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # type: ignore[import]
from isaaclab.envs import DirectRLEnv  # type: ignore[import]
from pxr import UsdGeom  # type: ignore[import]

if TYPE_CHECKING:
    from isaacsim.core.utils import prims as prim_utils  # type: ignore[import]
else:
    import importlib

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")

from .dofbot_pickplace_env_cfg import DofbotPickPlaceEnvCfg


class DofbotPickPlaceEnv(DirectRLEnv):
    """Dofbot pick-and-place 환경 구현.

    관측:
            - q(5), dq(5)
      - ee_pos(3), obj_pos(3), goal_pos(3)
      - grip_joint 위치(1)

    액션:
      - 6차원: 5개 관절 속도 + 1개 그리퍼 속도 (정규화 [-1, 1])
    """

    cfg: DofbotPickPlaceEnvCfg

    def __init__(
        self, cfg: DofbotPickPlaceEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve indices (preserve configured order)
        self._arm_dof_idx, _ = self.robot.find_joints(
            self.cfg.arm_dof_names, preserve_order=True
        )
        self._grip_dof_idx, _ = self.robot.find_joints(
            self.cfg.gripper_dof_name, preserve_order=True
        )
        # End-effector link index (for pose)
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.ee_link_name)

        # Shortcuts to buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Per-environment caches for object/goal positions (env-local frame)
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        # Tracks whether each environment currently holds the object so rewards persist while carrying
        self._grasp_state = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Placement constraints (front sector + safety radius) pulled from cfg
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

        # Configure joint drives after scene setup
        self._configure_joint_drives()

    # Scene ------------------------------------------------------------------
    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Create Object and Goal as RigidObjects for proper physics simulation
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal = RigidObject(self.cfg.goal_cfg)

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)
        self._hide_env_ground_planes()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register articulations and rigid objects
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal"] = self.goal

        # Light - 더 밝게
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _hide_env_ground_planes(self) -> None:
        """Hide environment ground planes cloned under each env namespace."""
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
        """Configure joint drive properties for VELOCITY control for all environments."""
        from pxr import UsdPhysics  # type: ignore

        # Get USD stage from simulation context
        stage = self.sim.stage

        # Configure drives for ALL environments (not just env_0)
        for env_idx in range(self.num_envs):
            for joint_name in self.cfg.arm_dof_names + [self.cfg.gripper_dof_name]:
                joint_path = f"/World/envs/env_{env_idx}/Robot/yahboom_dofbot/joints/{joint_name}"
                joint_prim = stage.GetPrimAtPath(joint_path)

                if joint_prim and joint_prim.IsValid():
                    # Get or create drive API
                    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

                    # Set drive parameters for VELOCITY control
                    drive_api.CreateDampingAttr(1000.0)
                    drive_api.CreateStiffnessAttr(0.0)
                    drive_api.CreateMaxForceAttr(1000.0)

    # Control/Step -----------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Use velocity control instead of effort
        # Split arm (5 joints) and gripper (1 joint)
        arm_velocities = self.actions[:, :5] * self.cfg.joint_velocity_scale
        grip_velocity = self.actions[:, 5:6] * self.cfg.grip_velocity_scale

        # Robot has 11 total joints, but we only control 6 (5 arm + 1 gripper)
        # Need to specify velocities for all joints - set uncontrolled joints to 0
        all_velocities = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        all_velocities[:, self._arm_dof_idx] = arm_velocities
        all_velocities[:, self._grip_dof_idx] = grip_velocity

        self.robot.set_joint_velocity_target(all_velocities)

    # Observations/Rewards/Dones --------------------------------------------
    def _get_observations(self) -> dict:
        # End-effector world position
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]].unsqueeze(-1)

        # Refresh cached object/goal positions (env-local frame)
        self._update_asset_states()
        obj_pos_w = self.obj_pos_w
        goal_pos_w = self.goal_pos_w

        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_dof_idx].reshape(self.num_envs, -1),
                self.joint_vel[:, self._arm_dof_idx].reshape(self.num_envs, -1),
                ee_pos_w,
                obj_pos_w,
                goal_pos_w,
                grip,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]]

        # Refresh cached object and goal states (convert back to world frame for distance metrics)
        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        # Distances and height
        d_reach = torch.linalg.norm(ee_pos_w - obj_pos_w, dim=-1)
        d_place = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)
        table_contact = self.cfg.table_height + 0.5 * self.cfg.object_size
        lift_height = torch.clamp(obj_pos_w[:, 2] - table_contact, min=0.0)
        lift_norm = torch.clamp(
            lift_height / max(self.cfg.lift_height_target, 1e-6), max=1.0
        )

        # Grasp-related masks
        close_mask = d_reach < self.cfg.pre_grasp_threshold
        grasp_candidate = (d_reach < self.cfg.grasp_threshold) & (
            grip < self.cfg.grip_closed_threshold
        )
        release_condition = (grip > self.cfg.grip_open_threshold) | (
            (lift_height < 0.005) & (~grasp_candidate)
        )
        self._grasp_state |= grasp_candidate
        self._grasp_state &= ~release_condition
        grasped_mask = self._grasp_state
        grasped_float = grasped_mask.float()

        # Gripper closure quality (0..1)
        closure_delta = torch.clamp(
            self.cfg.grip_closed_threshold - grip,
            min=0.0,
            max=self.cfg.grip_close_range,
        )
        closure_quality = closure_delta / max(self.cfg.grip_close_range, 1e-6)

        # Reward shaping terms
        reward = torch.full(
            (self.num_envs,), self.cfg.rew_scale_alive, device=self.device
        )
        reward += self.cfg.rew_scale_reach * d_reach
        reward += self.cfg.rew_scale_close * close_mask.float() * closure_quality
        reward += self.cfg.rew_scale_lift * grasped_float * lift_norm
        reward += self.cfg.rew_scale_goal_track * grasped_float * d_place
        reward += self.cfg.rew_penalty_open * torch.clamp(
            grip - self.cfg.grip_open_threshold, min=0.0
        )
        reward += (
            self.cfg.rew_bonus_grasp
            * (close_mask & (grip < self.cfg.grip_closed_threshold)).float()
        )
        reward += (
            self.cfg.rew_bonus_place
            * ((d_place < self.cfg.place_threshold) & grasped_mask).float()
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success as termination (place achieved)
        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        d_place = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)
        grasped_mask = self._grasp_state
        success = (d_place < self.cfg.place_threshold) & grasped_mask
        return success, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if isinstance(env_ids, torch.Tensor):
            env_tensor = env_ids.to(device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_tensor = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_tensor = torch.tensor(
                list(env_ids), device=self.device, dtype=torch.long
            )
        env_tensor = env_tensor.reshape(-1)
        env_count = env_tensor.shape[0]
        self._grasp_state[env_tensor] = False

        # Reset robot state to defaults + small noise
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        noise_pos = self.cfg.joint_pos_noise * (2.0 * torch.rand_like(joint_pos) - 1.0)
        noise_vel = self.cfg.joint_vel_noise * (2.0 * torch.rand_like(joint_vel) - 1.0)
        joint_pos = joint_pos + noise_pos
        joint_vel = joint_vel + noise_vel

        # Root pose per env
        default_root = self.robot.data.default_root_state[env_ids]
        default_root[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset object & goal positions using RigidObject API
        # Sample random positions in local frame
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

        goal_xy = self._sample_goal_xy(
            env_ids,
            obj_xy,
        )
        goal_z = torch.full((env_count, 1), self.cfg.goal_height, device=self.device)
        goal_pos_local = torch.cat((goal_xy, goal_z), dim=-1)

        # Convert to world frame
        obj_pos_w = obj_pos_local + self.scene.env_origins[env_ids]
        goal_pos_w = goal_pos_local + self.scene.env_origins[env_ids]

        # Create root state tensors for RigidObjects [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        object_state = torch.zeros((env_count, 13), device=self.device)
        object_state[:, :3] = obj_pos_w
        object_state[:, 3] = 1.0  # Identity quaternion (w = 1)

        goal_state = torch.zeros((env_count, 13), device=self.device)
        goal_state[:, :3] = goal_pos_w
        goal_state[:, 3] = 1.0

        # Keep local caches aligned with newly sampled poses
        self.obj_pos_w[env_tensor] = obj_pos_local
        self.goal_pos_w[env_tensor] = goal_pos_local

        # Write to simulation using RigidObject API
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

        # Normalize env_ids into a tensor on the current device for deterministic hashing
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
        env_tensor = env_tensor.reshape(-1)

        front_flag = front_only or getattr(self, "_front_only", False)

        # Base random component (common generator ensures reproducibility across runs)
        base = torch.rand((env_tensor.shape[0], 3), device=self.device)
        # Offset samples with hashed env-dependent phases to avoid perfectly correlated resets
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
            angles = (samples[:, 1] - 0.5) * math.pi  # [-pi/2, pi/2]
        else:
            angles = samples[:, 1] * 2.0 * math.pi  # [0, 2pi]

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
        self,
        env_ids: Sequence[int],
        obj_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Sample goal XY positions with a minimum separation from the object."""
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
                mask = torch.ones_like(
                    remaining_pos, dtype=torch.bool, device=self.device
                )

            if mask.any():
                accepted_pos = remaining_pos[mask]
                goal_xy[accepted_pos] = candidates[mask]
                remaining_pos = remaining_pos[~mask]
            attempts += 1

        # Fallback: if separation not satisfied for some envs, shift forward from object
        if remaining_pos.numel() > 0:
            obj_rem = obj_xy[remaining_pos]
            axis = self._front_axis
            lat_axis = 1 - axis
            low, high = self.cfg.goal_xy_range
            front_min = max(min_sep, 1e-4)

            if separation_tol <= 0.0:
                separation_tol = 0.12

            fallback = obj_rem.clone()
            # Encourage goals to land further away along front axis
            forward_offset = separation_tol + 0.05
            fallback[:, axis] = torch.clamp(
                obj_rem[:, axis] + forward_offset, min=front_min, max=high
            )

            # Lateral spread: push towards edges based on random sign
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
                fallback_fix = fallback[short_mask]
                fallback_fix[:, axis] = high
                fallback_fix[:, lat_axis] = torch.where(
                    obj_rem[short_mask, lat_axis] >= 0.0, low, high
                )
                fallback[short_mask] = fallback_fix

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
        """Refresh cached object and goal poses in the env-local frame."""
        torch.sub(
            self.object.data.root_pos_w, self.scene.env_origins, out=self.obj_pos_w
        )
        torch.sub(
            self.goal.data.root_pos_w, self.scene.env_origins, out=self.goal_pos_w
        )
