# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Tee-Align-Elfin-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.elfin_env_cfg:ElfinTeeAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ElfinTeeAlignPPORunnerCfg",

    },
)

gym.register(
    id="Isaac-Tee-Align-Elfin-Play-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.elfin_env_cfg:ElfinTeeAlignEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ElfinTeeAlignPPORunnerCfg",
    },
)

__all__ = ["ElfinTeeAlignEnvCfg", "ElfinTeeAlignEnvCfg_PLAY"]