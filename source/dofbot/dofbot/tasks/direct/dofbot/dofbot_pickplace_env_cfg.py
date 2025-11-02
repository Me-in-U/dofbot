# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DofbotPickPlaceEnvCfg(DirectRLEnvCfg):
    """Pick-and-place 작업을 위한 Dofbot 환경 설정."""

    # Env
    decimation = 2
    episode_length_s = 10.0

    # Spaces
    action_space = 6  # 5 arm joints + 1 gripper
    observation_space = 20  # [q(5), dq(5), ee_pos(3), obj_pos(3), goal_pos(3), grip(1)]
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot USD 경로
    # assets 폴더는 `source/dofbot/assets` 에 위치합니다.
    # 현재 파일 경로: source/dofbot/dofbot/tasks/direct/dofbot/dofbot_pickplace_env_cfg.py
    # 따라서 상위로 4단계 올라가야 합니다.
    _ASSETS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
    )
    # 기본 로봇 USD 후보들
    _LOCAL_USD = os.path.join(_ASSETS_DIR, "dofbot.usd")
    _DEFAULT_USD = os.path.join(_ASSETS_DIR, "dofbot.usda")  # 최후 수단
    # 저장소 루트의 dofbot_moveit 변환 산출물 (권장 후보)
    # assets 디렉터리에서 저장소 루트(DofBot-Issac-Sim)로 올라가는 상대 경로는 6단계
    _REPO_ROOT = os.path.normpath(
        os.path.join(_ASSETS_DIR, "..", "..", "..", "..", "..", "..")
    )
    _MOVEIT_USD_CANDIDATE = os.path.join(
        _REPO_ROOT, "dofbot_moveit", "urdf", "dofbot", "dofbot.usd"
    )

    # 외부(절대경로) 로봇 USD 경로를 환경변수로 오버라이드할 수 있음
    # 예: set DOFBOT_ROBOT_USD=C:\path\to\dofbot_moveit\urdf\dofbot\dofbot.usd
    _ENV_OVERRIDE = os.environ.get("DOFBOT_ROBOT_USD")

    _SOURCE = "fallback"
    if _ENV_OVERRIDE and os.path.exists(_ENV_OVERRIDE):
        robot_usd_path = _ENV_OVERRIDE
        _SOURCE = "env"
        print(
            f"[DofbotPickPlaceEnvCfg] Using robot asset from DOFBOT_ROBOT_USD: '{robot_usd_path}'"
        )
    elif os.path.exists(_DEFAULT_USD):
        # Prefer ASCII USD (usda) first to avoid hidden/broken internal references in binary usd
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(
            f"[DofbotPickPlaceEnvCfg] Using local ASCII robot asset: '{robot_usd_path}'"
        )
    elif os.path.exists(_LOCAL_USD):
        robot_usd_path = _LOCAL_USD
        _SOURCE = "local"
        print(
            f"[DofbotPickPlaceEnvCfg] Using local binary robot asset: '{robot_usd_path}'"
        )
    elif os.path.exists(_MOVEIT_USD_CANDIDATE):
        robot_usd_path = _MOVEIT_USD_CANDIDATE
        _SOURCE = "moveit"
        print(
            f"[DofbotPickPlaceEnvCfg] Using MoveIt-converted robot asset: '{robot_usd_path}'"
        )
    else:
        robot_usd_path = _DEFAULT_USD
        _SOURCE = "usda"
        print(
            f"[DofbotPickPlaceEnvCfg] Warning: Using fallback ASCII robot asset '{robot_usd_path}'. "
            "If articulation fails to initialize, set DOFBOT_ROBOT_USD to a valid robot USD."
        )

    # 아티큘레이션 루트 Prim 경로 설정
    # - Isaac Lab은 prim_path와 articulation_root_prim_path를 결합합니다.
    # - 따라서 여기에는 '/World/...' 전체 경로가 아니라 Robot 하위의 '자식 Prim 이름'만 넣어야 합니다.
    # - 환경변수 DOFBOT_ARTICULATION_ROOT_NAME 으로 이름을 지정할 수 있습니다.
    _ART_NAME_ENV = os.environ.get("DOFBOT_ARTICULATION_ROOT_NAME")
    if _ART_NAME_ENV:
        # 허용 입력:
        #  - 'child'
        #  - 'child/sub'
        #  - '/child/sub'
        #  - '/World/.../Robot/child/sub'
        val = _ART_NAME_ENV.strip()
        # 만약 절대 경로로 들어오면 '/Robot' 이후의 서브경로만 추출
        if "/Robot/" in val:
            val = val.split("/Robot/")[-1]
        # 참조 USD의 defaultPrim(예: 'yahboom_dofbot')는 prim_path('/World/.../Robot')에 합성되므로
        # '/yahboom_dofbot/base_link'와 같이 defaultPrim 세그먼트는 보통 제거되어야 합니다.
        # 사용자가 그런 경로를 넣은 경우 자동으로 한 단계 제거합니다.
        # 값이 비어있다면 자산 내부 트리의 루트 서브트리('/World/yahboom_dofbot')를 기본으로 사용
        norm = "/" + val.lstrip("/") if val else "/World/yahboom_dofbot"
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            # 첫 세그먼트가 defaultPrim 이름처럼 보이면 제거 (예: 'yahboom_dofbot')
            # 안전을 위해 유명한 후보만 제거합니다.
            if parts[0] in {"yahboom_dofbot", "dofbot"}:
                norm = "/" + "/".join(parts[1:])
                print(
                    f"[DofbotPickPlaceEnvCfg] Normalized articulation root by dropping defaultPrim segment: '{norm}'"
                )
        # 루트가 서브트리 노드('/World/yahboom_dofbot')만 가리키면 강체 매칭이 실패하므로
        # 관용적으로 루트 강체인 'base_link'를 자동 보정합니다.
        if norm.endswith("/yahboom_dofbot"):
            norm = norm + "/base_link"
            print(
                f"[DofbotPickPlaceEnvCfg] Auto-corrected articulation root to rigid body: '{norm}'"
            )
        articulation_root_prim_path = norm
        print(
            f"[DofbotPickPlaceEnvCfg] Using articulation root from env (normalized): '{articulation_root_prim_path}'"
        )
    elif _SOURCE in {"local", "usda", "moveit"}:
        # USD의 defaultPrim="World"이므로 USD 참조 시 World의 내용이 Robot prim에 합성됩니다.
        # Isaac Lab이 '/World/envs/env_*/Robot'에 이 USD를 참조로 추가하면:
        #   - USD 내부: /World/yahboom_dofbot/base_link
        #   - 합성 결과: /World/envs/env_*/Robot/yahboom_dofbot/base_link (World는 defaultPrim이라 생략됨)
        # articulation_root_prim_path는 절대 경로로 시작해야 하므로 '/yahboom_dofbot/base_link'
        articulation_root_prim_path = "/yahboom_dofbot/base_link"
        print(
            f"[DofbotPickPlaceEnvCfg] Using articulation root path: '{articulation_root_prim_path}'. "
            f"Full pattern: /World/envs/env_*/Robot{articulation_root_prim_path}"
        )
    else:
        articulation_root_prim_path = None

    # Robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "arm_joint.*": 0.0,
                "grip_joint": 0.0,
            },
        ),
        actuators={},  # Direct control - no actuator models needed
        articulation_root_prim_path=articulation_root_prim_path,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.0, replicate_physics=True
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

    # Action scaling - CHANGED to velocity control (effort not working)
    joint_velocity_scale = 1.0  # rad/s - joint velocity limits
    grip_velocity_scale = 0.5  # rad/s - gripper velocity limit

    # Object/Goal 배치 범위 (로봇 바로 앞, 매우 가깝게)
    table_height = 0.05  # 테이블 높이
    object_size = 0.03
    object_xy_range = (-0.08, 0.08)  # 로봇 바로 앞
    goal_xy_range = (-0.10, 0.10)  # 약간 더 넓은 범위
    goal_height = 0.12  # 로봇이 도달 가능한 높이

    # Rewards
    rew_scale_alive = 0.1
    rew_scale_reach = -1.0
    rew_scale_transport = -1.0
    rew_bonus_grasp = 2.0
    rew_bonus_place = 5.0

    # Success/termination
    grasp_threshold = 0.03
    place_threshold = 0.03

    # Reset randomization
    joint_pos_noise = 0.05
    joint_vel_noise = 0.05
