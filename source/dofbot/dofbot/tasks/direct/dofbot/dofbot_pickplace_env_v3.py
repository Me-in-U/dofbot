# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-Place V3: Aggressive Action & Simplified Curriculum
- V2 Problem: Only REACH works, no gripper/arm movement after
- V3 Solution: Larger action scale, gripper emphasis, 3-stage curriculum
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

from .dofbot_pickplace_env_cfg_v3 import DofbotPickPlaceEnvCfgV3


class DofbotPickPlaceEnvV3(DirectRLEnv):
    """
    Dofbot pick-and-place V3 환경 구현

    V2 → V3 주요 변경:
    1. **3-stage simplified curriculum** (5→3 stages)
       - Stage 0: REACH (EE → Object)
       - Stage 1: GRASP+LIFT (Close gripper + Lift)
       - Stage 2: PLACE (Transport + Release)
    
    2. **Aggressive action scaling**
       - Arm actions: 2.5x
       - Gripper actions: 4.0x (특히 강조!)
    
    3. **Gripper exploration bonus**
       - 그리퍼가 움직이면 보상
    
    4. **Relaxed detection thresholds**
       - Grasp: 0.03 (V2: 0.1)
       - Lift: 0.08m (V2: 0.10m)
       - Goal: 0.08m (V2: 0.05m)
    
    5. **No jerk penalty**
       - 급격한 움직임 허용

    관측 (23D):
        - q(5), dq(5)
        - ee_pos(3), obj_pos(3), goal_pos(3)
        - grip_joint(1), obj_vel(3)

    액션 (6D):
        - 5 arm joint velocities (scaled by 2.5)
        - 1 gripper velocity (scaled by 4.0!)
    """

    cfg: DofbotPickPlaceEnvCfgV3

    def __init__(
        self, cfg: DofbotPickPlaceEnvCfgV3, render_mode: str | None = None, **kwargs
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
        
        # V3: Gripper contact bodies (both finger tips)
        self._gripper_bodies_idx, _ = self.robot.find_bodies(
            self.cfg.gripper_contact_bodies, preserve_order=True
        )

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

        # V3: Simplified 3-stage tracking
        self._current_stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Stages: 0=REACH, 1=GRASP+LIFT, 2=PLACE

        # V3: Track gripper movement for exploration bonus
        self._prev_gripper_pos = torch.zeros(self.num_envs, device=self.device)

        # Placement constraints
        axis = getattr(self.cfg, "placement_front_axis", "x")
        if isinstance(axis, str):
            axis = axis.lower()
            if axis not in ("x", "y"):
                raise ValueError(f"placement_front_axis must be 'x' or 'y', got {axis}")
        if axis == "x":
            self._placement_x_only = True
            self._placement_y_only = False
        else:
            self._placement_x_only = False
            self._placement_y_only = True

        self._front_only = getattr(self.cfg, "placement_front_only", True)
        self._object_min_radius = getattr(self.cfg, "object_min_radius", 0.0)
        self._goal_min_radius = getattr(self.cfg, "goal_min_radius", 0.0)

        # Print debug info
        print("[V3] ========================================")
        print("[V3] Dofbot Pick-and-Place V3 Environment")
        print("[V3] ========================================")
        print(f"[V3] Number of environments: {self.num_envs}")
        print(f"[V3] Action scaling - Arm: {self.cfg.action_scale}x")
        print(f"[V3] Action scaling - Gripper: {self.cfg.gripper_action_scale}x")
        print(f"[V3] Curriculum stages: {self.cfg.num_stages}")
        print(f"[V3] Gripper structure: grip_joint controls rlink2/llink2")
        print(f"[V3] EE link: {self.cfg.ee_link_name} (actual finger tip)")
        print(f"[V3] Contact bodies: {self.cfg.gripper_contact_bodies}")
        print(f"[V3] Grasp threshold: {self.cfg.grasp_contact_threshold}")
        print(f"[V3] Lift threshold: {self.cfg.lift_height_threshold}m")
        print(f"[V3] Goal tolerance: {self.cfg.goal_tolerance}m")
        print("[V3] ========================================")

    # Setup ---------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal = RigidObject(self.cfg.goal_cfg)

        # V3: Clone environments first (like V2)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # V3: Register assets after cloning
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal"] = self.goal

        # V3: Brighter light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    # Pre-physics ---------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        V3: Separate scaling for arm (2.5x) and gripper (4.0x)
        """
        # Split actions
        arm_actions = actions[:, :5]  # First 5 for arm joints
        gripper_actions = actions[:, 5:6]  # Last 1 for gripper

        # V3: Different scaling for arm vs gripper
        arm_scaled = arm_actions * self.cfg.action_scale  # 2.5x
        gripper_scaled = gripper_actions * self.cfg.gripper_action_scale  # 4.0x

        # Create full-size velocity target tensor (11 joints total)
        all_velocities = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.device
        )
        all_velocities[:, self._arm_dof_idx] = arm_scaled
        all_velocities[:, self._grip_dof_idx] = gripper_scaled

        self.actions = actions.clone()
        self.robot.set_joint_velocity_target(all_velocities)

    # Post-physics --------------------------------------------------------
    def _apply_action(self):
        pass

    def _update_asset_states(self):
        """캐싱된 object/goal 위치 업데이트"""
        self.obj_pos_w = self.object.data.root_pos_w - self.scene.env_origins
        self.goal_pos_w = self.goal.data.root_pos_w - self.scene.env_origins

    def _get_observations(self) -> dict:
        """관측 생성 (V2와 동일)"""
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]].unsqueeze(-1)

        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

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
        V3: Simplified 3-stage reward with gripper exploration bonus
        
        Stage 0: REACH (EE → Object)
        Stage 1: GRASP+LIFT (Close gripper + Lift object)
        Stage 2: PLACE (Transport + Release at goal)
        """
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]]

        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        # V3: Use actual gripper finger tips for distance calculation
        # Get both finger tip positions
        rlink2_pos = self.robot.data.body_pos_w[:, self._gripper_bodies_idx[0]]  # rlink2
        llink2_pos = self.robot.data.body_pos_w[:, self._gripper_bodies_idx[1]]  # llink2
        
        # Calculate center between two finger tips
        gripper_center = (rlink2_pos + llink2_pos) / 2.0
        
        # Distances
        d_ee_obj = torch.linalg.norm(ee_pos_w - obj_pos_w, dim=-1)  # EE (rlink2) to object
        d_gripper_obj = torch.linalg.norm(gripper_center - obj_pos_w, dim=-1)  # Gripper center to object
        d_obj_goal = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)

        # Object height above table
        table_height = 0.05  # Assume table at 5cm
        obj_height = torch.clamp(obj_pos_w[:, 2] - table_height, min=0.0)

        # V3: Improved grasp detection using gripper center
        gripper_closed = grip < self.cfg.grasp_close_joint_threshold  # -0.7
        near_object = d_gripper_obj < self.cfg.grasp_contact_threshold  # 0.03 (양쪽 finger tip 중심 기준)

        # Update grasp state
        grasp_candidate = near_object & gripper_closed
        self._grasp_state |= grasp_candidate

        # V3: Relaxed stage conditions
        reached = d_ee_obj < 0.10  # 10cm 이내면 reached
        grasped = self._grasp_state
        lifted = obj_height > self.cfg.lift_height_threshold  # 0.08m
        at_goal = d_obj_goal < self.cfg.goal_tolerance  # 0.08m

        # Update stage tracking (0=REACH, 1=GRASP+LIFT, 2=PLACE)
        self._current_stage = torch.where(reached, 1, 0)
        self._current_stage = torch.where(grasped, 1, self._current_stage)
        self._current_stage = torch.where(grasped & lifted, 2, self._current_stage)

        # Initialize reward
        reward = torch.zeros(self.num_envs, device=self.device)

        # ===== Stage 0: REACH =====
        # Distance reward (negative, so closer = higher)
        reward -= self.cfg.rew_reach_dist_scale * d_ee_obj

        # Reaching bonus
        reward += 5.0 * reached.float()

        # ===== Stage 1: GRASP+LIFT =====
        if reached.any():
            # V3: 그리퍼 닫기 강력히 보상
            gripper_closure = torch.clamp(
                (-0.7 - grip) / 1.2,  # -0.7이 목표, 1.2는 range
                min=0.0,
                max=1.0,
            )
            reward += 10.0 * reached.float() * gripper_closure

        # Grasp success bonus
        reward += self.cfg.rew_grasp_success_bonus * grasped.float()

        # Lift reward
        if grasped.any():
            lift_progress = torch.clamp(obj_height / self.cfg.lift_height_threshold, max=1.0)
            reward += 10.0 * grasped.float() * lift_progress

        # Lift success bonus
        reward += self.cfg.rew_lift_success_bonus * (grasped & lifted).float()

        # ===== Stage 2: PLACE =====
        if (grasped & lifted).any():
            # Transport reward
            reward -= 5.0 * (grasped & lifted).float() * d_obj_goal

            # At goal bonus
            reward += self.cfg.rew_place_success_bonus * (grasped & lifted & at_goal).float()

        # ===== V3: Gripper exploration bonus =====
        gripper_movement = torch.abs(grip - self._prev_gripper_pos)
        reward += self.cfg.rew_gripper_exploration_bonus * gripper_movement
        self._prev_gripper_pos = grip.clone()

        # ===== V3: Minimal penalties =====
        # Small action penalty (허용적)
        action_magnitude = torch.sum(torch.abs(self.actions), dim=-1)
        reward -= self.cfg.rew_action_penalty_scale * action_magnitude

        # V3: No jerk penalty (제거됨)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # V3: Success = object at goal while lifted (grasped 여부 무관)
        self._update_asset_states()
        obj_pos_w = self.obj_pos_w + self.scene.env_origins
        goal_pos_w = self.goal_pos_w + self.scene.env_origins

        d_place = torch.linalg.norm(obj_pos_w - goal_pos_w, dim=-1)
        obj_height = torch.clamp(obj_pos_w[:, 2] - 0.05, min=0.0)
        
        success = (d_place < self.cfg.goal_tolerance) & (obj_height > 0.03)

        return success, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_tensor = self._to_env_tensor(env_ids)
        env_count = env_tensor.shape[0]

        # Reset tracking
        self._grasp_state[env_tensor] = False
        self._current_stage[env_tensor] = 0
        self._prev_gripper_pos[env_tensor] = 0.0

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # V3: Small noise for exploration
        joint_pos += 0.05 * (2.0 * torch.rand_like(joint_pos) - 1.0)
        joint_vel += 0.05 * (2.0 * torch.rand_like(joint_vel) - 1.0)

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
        obj_z = torch.full((env_count, 1), 0.07, device=self.device)  # 7cm height
        obj_pos_local = torch.cat((obj_xy, obj_z), dim=-1)

        goal_xy = self._sample_goal_xy(env_ids, obj_xy)
        goal_z = torch.full((env_count, 1), 0.03, device=self.device)  # 3cm height
        goal_pos_local = torch.cat((goal_xy, goal_z), dim=-1)

        obj_pos_w = obj_pos_local + self.scene.env_origins[env_ids]
        goal_pos_w = goal_pos_local + self.scene.env_origins[env_ids]

        object_state = torch.zeros((env_count, 13), device=self.device)
        object_state[:, :3] = obj_pos_w
        object_state[:, 3] = 1.0  # quaternion w

        goal_state = torch.zeros((env_count, 13), device=self.device)
        goal_state[:, :3] = goal_pos_w
        goal_state[:, 3] = 1.0

        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)
        self.goal.write_root_pose_to_sim(goal_state[:, :7], env_ids)

    # Helpers -------------------------------------------------------------
    def _to_env_tensor(self, env_ids):
        if isinstance(env_ids, slice):
            start = env_ids.start or 0
            stop = env_ids.stop or self.num_envs
            step = env_ids.step or 1
            return torch.arange(start, stop, step, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _sample_xy(
        self,
        env_ids,
        xy_range: tuple[float, float],
        min_radius: float = 0.0,
        front_only: bool = True,
    ):
        """샘플링 헬퍼"""
        n = len(env_ids)
        lo, hi = xy_range
        xy = torch.zeros((n, 2), device=self.device)

        if self._placement_x_only:
            x = torch.rand(n, device=self.device) * (hi - lo) + lo
            if front_only and min_radius > 0:
                x = torch.clamp(x, min=min_radius)
            xy[:, 0] = x
        elif self._placement_y_only:
            y = torch.rand(n, device=self.device) * (hi - lo) + lo
            if front_only and min_radius > 0:
                y = torch.abs(y)
                y = torch.where(y < min_radius, min_radius + torch.rand_like(y) * (hi - min_radius), y)
            xy[:, 1] = y
        else:
            xy = torch.rand((n, 2), device=self.device) * (hi - lo) + lo

        return xy

    def _sample_goal_xy(self, env_ids, obj_xy: torch.Tensor):
        """Goal 위치 샘플링 (object와 최소 거리 유지)"""
        n = len(env_ids)
        min_sep = self.cfg.goal_object_min_separation
        lo, hi = self.cfg.goal_xy_range

        goal_xy = torch.zeros((n, 2), device=self.device)
        for i in range(n):
            for _ in range(100):
                candidate = self._sample_xy(
                    [0],
                    self.cfg.goal_xy_range,
                    min_radius=self.cfg.goal_min_radius,
                    front_only=self._front_only,
                )
                dist = torch.linalg.norm(candidate[0] - obj_xy[i])
                if dist >= min_sep:
                    goal_xy[i] = candidate[0]
                    break
            else:
                goal_xy[i] = obj_xy[i] + torch.tensor(
                    [min_sep, 0.0], device=self.device
                )

        return goal_xy
