from typing import Dict, Any

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs.mdp.observations import (
    root_pos_w,
    root_quat_w,
    joint_pos_rel as base_joint_pos_rel,
    joint_vel_rel as base_joint_vel_rel,
    last_action as base_last_action,
    generated_commands as base_generated_commands,
    image as base_image
)
from isaaclab.managers import SceneEntityCfg, CommandTerm
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from .commands import TeeAlignCommand


def last_action(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the last action applied to the environment."""
    return base_last_action(env)


def joint_pos_rel(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the current joint positions normalized to [-1, 1]."""
    return base_joint_pos_rel(env, SceneEntityCfg("robot"))


def joint_vel_rel(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the current joint velocities normalized to [-1, 1]."""
    return base_joint_vel_rel(env, SceneEntityCfg("robot"))


def tee_object_pos(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the position of the T-shaped object in world frame."""
    return root_pos_w(env, SceneEntityCfg("tee_object"))


def tee_object_rot(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the orientation of the T-shaped object in world frame."""
    return root_quat_w(env, False, SceneEntityCfg("tee_object"))


def tee_marker_pos(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the position of the T-shaped marker in world frame."""
    return root_pos_w(env, SceneEntityCfg("tee_marker"))


def tee_marker_rot(env: ManagerBasedEnv, _: Any) -> torch.Tensor:
    """Get the orientation of the T-shaped marker in world frame."""
    return root_quat_w(env, False, SceneEntityCfg("tee_marker"))


def image(env: ManagerBasedEnv, params: Dict[str, Any]) -> torch.Tensor:
    """Get the camera image."""
    return base_image(env, params["sensor_cfg"], params["data_type"], normalize=params["normalize"])


def generated_commands(env: ManagerBasedRLEnv, params: Dict[str, Any]) -> torch.Tensor:
    """Get the generated commands from the command term."""
    command_term: CommandTerm = env.command_manager.get_term(params["command_name"])
    return command_term.command


def goal_quat_diff(env: ManagerBasedRLEnv, params: Dict[str, Any]) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[params["asset_cfg"].name]
    command_term: CommandTerm = env.command_manager.get_term(params["command_name"])

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if params["make_quat_unique"] else quat 