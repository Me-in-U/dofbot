# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Dofbot-PickPlace-Direct-v0",
    entry_point=f"{__name__}.dofbot_pickplace_env:DofbotPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_pickplace_env_cfg:DofbotPickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pickplace_cfg.yaml",
    },
)

gym.register(
    id="Dofbot-PickPlace-Direct-v2",
    entry_point=f"{__name__}.dofbot_pickplace_env_v2:DofbotPickPlaceEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_pickplace_env_cfg_v2:DofbotPickPlaceEnvCfgV2",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pickplace_v2_cfg.yaml",
    },
)

gym.register(
    id="Dofbot-PickPlace-Direct-v3",
    entry_point=f"{__name__}.dofbot_pickplace_env_v3:DofbotPickPlaceEnvV3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_pickplace_env_cfg_v3:DofbotPickPlaceEnvCfgV3",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pickplace_v3_cfg.yaml",
    },
)
