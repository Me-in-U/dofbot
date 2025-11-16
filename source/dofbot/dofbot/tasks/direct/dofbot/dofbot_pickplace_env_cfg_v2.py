# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-Place V2: Curriculum Learning with Stage-based Rewards
- Improved grasp detection (contact-based)
- Clearer reward shaping per stage
- Better exploration strategy
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DofbotSceneCfg(InteractiveSceneCfg):
    """Scene definition with a shared ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.4, 0.4, 0.4), size=(100.0, 100.0), visible=True
        ),
        collision_group=-1,
    )


@configclass
class DofbotPickPlaceEnvCfgV2(DirectRLEnvCfg):
    """
    Pick-and-Place V2 환경 설정

    개선 사항:
    - Curriculum learning with clear stages
    - Contact-based grasp detection
    - Simplified reward structure
    - Better exploration
    """

    # Env
    decimation = 2
    episode_length_s = 15.0  # 더 긴 에피소드 (학습 초기 exploration)

    # Spaces
    action_space = 6  # 5 arm joints + 1 gripper
    observation_space = (
        23  # [q(5), dq(5), ee_pos(3), obj_pos(3), goal_pos(3), grip(1), obj_vel(3)]
    )
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot USD 경로 (V1과 동일)
    _ASSETS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
    )
    _LOCAL_USD = os.path.join(_ASSETS_DIR, "dofbot.usd")
    _DEFAULT_USD = os.path.join(_ASSETS_DIR, "dofbot.usda")
    _REPO_ROOT = os.path.normpath(
        os.path.join(_ASSETS_DIR, "..", "..", "..", "..", "..", "..")
    )
    _MOVEIT_USD_CANDIDATE = os.path.join(
        _REPO_ROOT, "dofbot_moveit", "urdf", "dofbot", "dofbot.usd"
    )
    _ENV_OVERRIDE = os.environ.get("DOFBOT_ROBOT_USD")

    _SOURCE = "fallback"
    if _ENV_OVERRIDE and os.path.exists(_ENV_OVERRIDE):
        robot_usd_path = _ENV_OVERRIDE
        _SOURCE = "env"
        print(f"[V2] Using robot asset from DOFBOT_ROBOT_USD: '{robot_usd_path}'")
    elif os.path.exists(_DEFAULT_USD):
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(f"[V2] Using local ASCII robot asset: '{robot_usd_path}'")
    elif os.path.exists(_LOCAL_USD):
        robot_usd_path = _LOCAL_USD
        _SOURCE = "local"
        print(f"[V2] Using local binary robot asset: '{robot_usd_path}'")
    elif os.path.exists(_MOVEIT_USD_CANDIDATE):
        robot_usd_path = _MOVEIT_USD_CANDIDATE
        _SOURCE = "moveit"
        print(f"[V2] Using MoveIt-converted robot asset: '{robot_usd_path}'")
    else:
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(f"[V2] Warning: Using fallback ASCII robot asset '{robot_usd_path}'")

    # Articulation root
    _ART_NAME_ENV = os.environ.get("DOFBOT_ARTICULATION_ROOT_NAME")
    if _ART_NAME_ENV:
        val = _ART_NAME_ENV.strip()
        if "/Robot/" in val:
            val = val.split("/Robot/")[-1]
        norm = "/" + val.lstrip("/") if val else "/World/yahboom_dofbot"
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            if parts[0] in {"yahboom_dofbot", "dofbot"}:
                norm = "/" + "/".join(parts[1:])
                print(f"[V2] Normalized articulation root: '{norm}'")
        if norm.endswith("/yahboom_dofbot"):
            norm = norm + "/base_link"
            print(f"[V2] Auto-corrected articulation root to: '{norm}'")
        articulation_root_prim_path = norm
        print(f"[V2] Using articulation root from env: '{articulation_root_prim_path}'")
    elif _SOURCE in {"local", "usda", "moveit"}:
        articulation_root_prim_path = "/yahboom_dofbot/base_link"
        print(f"[V2] Using articulation root path: '{articulation_root_prim_path}'")
    else:
        articulation_root_prim_path = None

    # Robot - 더 강한 actuator
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "arm_joint.*": 0.0,
                "grip_joint": -0.5,  # 초기 그리퍼 약간 닫힌 상태
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[1-5]"],
                stiffness=0.0,
                damping=50.0,  # 더 높은 damping
                effort_limit_sim=50.0,  # 더 높은 effort limit
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["grip_joint"],
                stiffness=0.0,
                damping=15.0,  # 더 높은 damping
                effort_limit_sim=30.0,  # 더 높은 effort limit
            ),
        },
        articulation_root_prim_path=articulation_root_prim_path,
    )

    # Object - 더 가벼운 물체 (grasp 용이)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.025, 0.025, 0.025),  # 더 작은 크기
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),  # 더 가벼움
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.2,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # Goal
    goal_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                emissive_color=(0.0, 0.5, 0.0),
                metallic=0.0,
                roughness=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.15)),
    )

    # Scene
    scene: DofbotSceneCfg = DofbotSceneCfg(
        num_envs=256, env_spacing=1.0, replicate_physics=True
    )

    # Joint/Link names
    arm_dof_names = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
    ]
    gripper_dof_name = "grip_joint"
    ee_link_name = "arm_link5"

    # Action scaling
    joint_velocity_scale = 1.5  # 더 빠른 움직임
    grip_velocity_scale = 0.8

    # Object/Goal 배치 (V1과 동일한 범위로 수정)
    table_height = 0.05
    object_size = 0.025  # V2: 더 작은 물체 (grasp 용이)
    object_xy_range = (-0.3, 0.3)  # V1과 동일
    goal_xy_range = (-0.3, 0.3)  # V1과 동일
    goal_height = 0.12
    object_min_radius = 0.15
    goal_min_radius = 0.25  # V1과 동일
    placement_front_only = True
    placement_front_axis = "y"  # V1과 동일: 로봇 정면(+Y)에만 배치
    goal_object_min_separation = 0.30  # V1과 동일

    # ========== V2: Curriculum Learning Rewards ==========
    # Stage 1: Reach (EE가 물체에 가까이 가도록)
    rew_stage1_reach = -2.0  # dense distance penalty
    rew_stage1_bonus = 3.0  # reach threshold 도달 시 bonus

    # Stage 2: Grasp (물체를 잡도록)
    rew_stage2_close_gripper = 2.0  # 그리퍼 닫기 보상
    rew_stage2_grasp_bonus = 5.0  # 실제 grasp 성공 시 큰 보상
    rew_stage2_hold_penalty = -0.5  # 그리퍼를 열고 있으면 패널티

    # Stage 3: Lift (물체를 들어올리도록)
    rew_stage3_lift = 4.0  # 높이에 비례한 보상
    rew_stage3_bonus = 3.0  # 목표 높이 도달 시 bonus

    # Stage 4: Transport (물체를 목표로 이동)
    rew_stage4_transport = -1.5  # 물체-목표 거리 패널티
    rew_stage4_progress = 2.0  # 목표에 가까워질수록 보상

    # Stage 5: Place (물체를 목표에 놓기)
    rew_stage5_place_bonus = 10.0  # 최종 성공 시 큰 보상

    # General penalties
    rew_penalty_action = -0.01  # action magnitude penalty
    rew_penalty_jerk = -0.01  # action smoothness penalty
    rew_alive = 0.05  # alive bonus

    # Thresholds
    reach_threshold = 0.06  # EE가 물체에 충분히 가까움
    grasp_threshold = 0.04  # 물체를 잡았다고 판단
    lift_threshold = 0.08  # 테이블에서 충분히 들어올림
    place_threshold = 0.04  # 목표에 충분히 가까움

    # Gripper thresholds
    grip_closed_threshold = -0.4  # 그리퍼가 닫힌 것으로 간주
    grip_open_threshold = -0.1  # 그리퍼가 열린 것으로 간주

    # Success/termination
    success_threshold = 0.03  # 최종 성공 판정 거리

    # Reset randomization
    joint_pos_noise = 0.1  # 더 큰 초기 randomization
    joint_vel_noise = 0.1
