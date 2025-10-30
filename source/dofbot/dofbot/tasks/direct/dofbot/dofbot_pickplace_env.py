# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

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

        # Resolve indices
        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_dof_names)
        self._grip_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_dof_name)
        # End-effector link index (for pose)
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.ee_link_name)

        # Shortcuts to buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Placeholders for object/goal
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

    # Scene ------------------------------------------------------------------
    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Visual helpers: goal marker (sphere) and object (cube)
        # Note: 스포너는 소스 env에만 생성 후 clone_environments로 복제됨
        self._spawn_pickplace_assets()

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register
        self.scene.articulations["robot"] = self.robot

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_pickplace_assets(self):
        # Object cube
        cube_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.object_size, self.cfg.object_size, self.cfg.object_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        )
        # Spawn under source env so that scene.clone_environments can replicate per env
        cube_cfg.func("/World/envs/env_0/Object", cube_cfg)

        # Goal sphere (visual only)
        goal_cfg = sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=None,
        )
        goal_cfg.func("/World/envs/env_0/Goal", goal_cfg)

    # Control/Step -----------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Split arm and gripper
        arm_actions = self.actions[:, :5] * self.cfg.joint_vel_scale
        grip_action = self.actions[:, 5:6] * self.cfg.grip_vel_scale
        # Apply joint velocity targets
        self.robot.set_joint_velocity_target(arm_actions, joint_ids=self._arm_dof_idx)
        self.robot.set_joint_velocity_target(grip_action, joint_ids=self._grip_dof_idx)

    # Observations/Rewards/Dones --------------------------------------------
    def _get_observations(self) -> dict:
        # End-effector world position
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]].unsqueeze(-1)
        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_dof_idx].reshape(self.num_envs, -1),
                self.joint_vel[:, self._arm_dof_idx].reshape(self.num_envs, -1),
                ee_pos_w,
                self.obj_pos_w,
                self.goal_pos_w,
                grip,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0]]
        grip = self.joint_pos[:, self._grip_dof_idx[0]]

        # Distances
        d_reach = torch.linalg.norm(ee_pos_w - self.obj_pos_w, dim=-1)
        d_place = torch.linalg.norm(self.obj_pos_w - self.goal_pos_w, dim=-1)

        # Grasp heuristic
        grasped = (d_reach < self.cfg.grasp_threshold) & (
            grip < -0.2
        )  # closed and close

        reward = (
            self.cfg.rew_scale_alive
            + self.cfg.rew_scale_reach * d_reach
            + self.cfg.rew_scale_transport * d_place * grasped.float()
            + self.cfg.rew_bonus_grasp * (d_reach < self.cfg.grasp_threshold).float()
            + self.cfg.rew_bonus_place
            * ((d_place < self.cfg.place_threshold) & grasped).float()
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success as termination (place achieved)
        d_place = torch.linalg.norm(self.obj_pos_w - self.goal_pos_w, dim=-1)
        grip = self.joint_pos[:, self._grip_dof_idx[0]]
        d_reach = torch.linalg.norm(
            self.robot.data.body_pos_w[:, self._ee_body_idx[0]] - self.obj_pos_w, dim=-1
        )
        grasped = (d_reach < self.cfg.grasp_threshold) & (grip < -0.2)
        success = (d_place < self.cfg.place_threshold) & grasped
        return success, time_out

    # Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

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

        # Reset object & goal positions (world)
        # Object on table (z = table_height + half size)
        obj_xy = self._sample_xy(env_ids, self.cfg.object_xy_range)
        obj_z = torch.full(
            (len(env_ids), 1),
            self.cfg.table_height + 0.5 * self.cfg.object_size,
            device=self.device,
        )
        obj_pos = torch.cat((obj_xy, obj_z), dim=-1)

        # Goal above table
        goal_xy = self._sample_xy(env_ids, self.cfg.goal_xy_range)
        goal_z = torch.full((len(env_ids), 1), self.cfg.goal_height, device=self.device)
        goal_pos = torch.cat((goal_xy, goal_z), dim=-1)

        # Add env origins
        obj_pos = obj_pos + self.scene.env_origins[env_ids]
        goal_pos = goal_pos + self.scene.env_origins[env_ids]

        self.obj_pos_w[env_ids] = obj_pos
        self.goal_pos_w[env_ids] = goal_pos

        # Move prims (visual markers) per env
        ident_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        for i, env_id in enumerate(env_ids):
            env_idx = int(env_id)
            self._set_prim_world_pose(
                f"/World/envs/env_{env_idx}/Object", obj_pos[i], ident_quat
            )
            self._set_prim_world_pose(
                f"/World/envs/env_{env_idx}/Goal", goal_pos[i], ident_quat
            )

    # Utils ---------------------------------------------------------------
    def _sample_xy(
        self, env_ids: Sequence[int], xy_range: tuple[float, float]
    ) -> torch.Tensor:
        low, high = xy_range
        xy = low + (high - low) * torch.rand((len(env_ids), 2), device=self.device)
        return xy

    def _set_prim_world_pose(
        self, prim_path: str, pos: torch.Tensor, quat_wxyz: torch.Tensor
    ) -> None:
        """Set a prim's world pose using USD Xform ops.

        Note: quat expected as [w, x, y, z].
        """
        try:
            import omni.usd
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return
            xformable = UsdGeom.Xformable(prim)
            # Clear existing ops order and set translate + orient
            # Use dedicated ops to avoid stacking transforms over time
            # Create ops (will reuse if they already exist with same opSuffix)
            translate_op = None
            orient_op = None
            # Reuse existing ops if present
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = xformable.AddOrientOp()

            translate_op.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
            orient_op.Set(
                Gf.Quatf(
                    float(quat_wxyz[0]),
                    Gf.Vec3f(
                        float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
                    ),
                )
            )
            xformable.SetXformOpOrder([translate_op, orient_op])
        except Exception:
            # Silently ignore pose set failures (visual-only helper)
            pass
