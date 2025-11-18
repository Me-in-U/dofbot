# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pick-and-Place V3: Aggressive Action & Simplified Curriculum
- V2 문제: REACH만 되고 그 이후 gripper/arm 움직임 없음
- V3 해결: 더 큰 action scale, gripper 강조, 3-stage curriculum
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
class DofbotPickPlaceEnvCfgV3(DirectRLEnvCfg):
    """
    Pick-and-Place V3 환경 설정

    V2 → V3 주요 개선:
    1. Action scale 2.0x (더 큰 움직임)
    2. Gripper action 별도 강조 (3.0x scaling)
    3. Simplified 3-stage curriculum (REACH → GRASP+LIFT → PLACE)
    4. Gripper exploration bonus 추가
    5. Relaxed grasp detection (threshold 0.05)
    6. Action jerk penalty 제거
    """

    # Env
    decimation = 2
    episode_length_s = 15.0

    # Spaces
    action_space = 6  # 5 arm joints + 1 gripper
    observation_space = 23  # Same as V2
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot USD 경로 (V2와 동일)
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
        print(f"[V3] Using robot asset from DOFBOT_ROBOT_USD: '{robot_usd_path}'")
    elif os.path.exists(_DEFAULT_USD):
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(f"[V3] Using local ASCII robot asset: '{robot_usd_path}'")
    elif os.path.exists(_LOCAL_USD):
        robot_usd_path = _LOCAL_USD
        _SOURCE = "local"
        print(f"[V3] Using local binary robot asset: '{robot_usd_path}'")
    elif os.path.exists(_MOVEIT_USD_CANDIDATE):
        robot_usd_path = _MOVEIT_USD_CANDIDATE
        _SOURCE = "moveit"
        print(f"[V3] Using MoveIt-converted robot asset: '{robot_usd_path}'")
    else:
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(f"[V3] Warning: Using fallback ASCII robot asset '{robot_usd_path}'")

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
                print(f"[V3] Normalized articulation root: '{norm}'")
        if norm.endswith("/yahboom_dofbot"):
            norm = norm + "/base_link"
            print(f"[V3] Auto-corrected articulation root to: '{norm}'")
        articulation_root_prim_path = norm
        print(f"[V3] Using articulation root from env: '{articulation_root_prim_path}'")
    elif _SOURCE in {"local", "usda", "moveit"}:
        articulation_root_prim_path = "/yahboom_dofbot/base_link"
        print(f"[V3] Using articulation root path: '{articulation_root_prim_path}'")
    else:
        articulation_root_prim_path = None

    # Robot - V3: 더욱 강한 actuator + 높은 effort limit
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "arm_joint.*": 0.0,
                "grip_joint": -0.3,  # V3: 그리퍼 더 열린 상태로 시작
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[1-5]"],
                stiffness=0.0,
                damping=60.0,  # V3: 더욱 높은 damping (빠른 반응)
                effort_limit_sim=80.0,  # V3: 훨씬 높은 effort limit
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["grip_joint"],
                stiffness=0.0,
                damping=20.0,  # V3: 더욱 높은 gripper damping
                effort_limit_sim=50.0,  # V3: 훨씬 높은 gripper effort
            ),
        },
        articulation_root_prim_path=articulation_root_prim_path,
    )

    # Object - V3: 매우 가벼운 물체 (grasp 매우 용이)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, disable_gravity=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),  # V3: 20g (매우 가벼움)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.05)),
    )

    # Goal marker
    goal_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalMarker",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), metallic=0.0, opacity=0.5
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.02)),
    )

    # Scene
    scene: InteractiveSceneCfg = DofbotSceneCfg(num_envs=4096, env_spacing=2.0)

    # Robot joint names
    arm_dof_names = [
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
    ]
    gripper_dof_name = ["grip_joint"]  # grip_joint controls rlink/llink gripper
    ee_link_name = "rlink2"  # V3: Actual gripper finger tip (right side)
    
    # V3: Gripper contact bodies (both finger tips for grasp detection)
    gripper_contact_bodies = ["rlink2", "llink2"]  # Both left and right finger tips

    # V3: Action scaling - 더 큰 움직임
    action_scale = 2.5  # V2: 1.0 → V3: 2.5 (2.5배 증가)
    gripper_action_scale = 4.0  # V3: 그리퍼는 4배로 더욱 강조!

    # V3: Simplified 3-stage curriculum
    # Stage 0: REACH (EE → Object)
    # Stage 1: GRASP+LIFT (Close gripper + Lift object)
    # Stage 2: PLACE (Transport + Release at goal)
    num_stages = 3

    # V3: Relaxed grasp detection
    grasp_contact_threshold = 0.03  # V2: 0.1 → V3: 0.03 (훨씬 쉽게 grasp 감지)
    grasp_close_joint_threshold = -0.7  # V3: 그리퍼가 70% 닫히면 grasp

    # V3: Relaxed lift detection
    lift_height_threshold = 0.08  # V2: 0.10 → V3: 0.08 (8cm면 lift 성공)

    # V3: Relaxed placement detection
    goal_tolerance = 0.08  # V2: 0.05 → V3: 0.08 (8cm 이내면 성공)

    # Reward weights - V3: 그리퍼 움직임 강조
    rew_reach_dist_scale = 5.0  # Same as V2
    rew_grasp_success_bonus = 15.0  # V2: 10.0 → V3: 15.0
    rew_lift_success_bonus = 20.0  # V2: 15.0 → V3: 20.0
    rew_place_success_bonus = 50.0  # V2: 30.0 → V3: 50.0
    rew_gripper_exploration_bonus = 2.0  # V3: 새로운! 그리퍼 움직임 보상
    rew_action_penalty_scale = 0.001  # V2: 0.01 → V3: 0.001 (10배 감소, 큰 움직임 허용)
    rew_jerk_penalty_scale = 0.0  # V3: 제거! (급격한 움직임 허용)

    # Placement constraints
    placement_front_axis = "y"  # V2에서 수정된 값 유지
    object_xy_range = (-0.3, 0.3)  # V2와 동일
    goal_xy_range = (-0.3, 0.3)  # V2와 동일
    goal_min_radius = 0.25  # V2와 동일
    goal_object_min_separation = 0.30  # V2와 동일
