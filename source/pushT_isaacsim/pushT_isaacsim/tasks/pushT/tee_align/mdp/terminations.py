import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length

def tee_aligned(
    env: "ManagerBasedRLEnv",
    distance_threshold: float = 0.02,
    angle_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    command_name: str = "tee_pose",
) -> torch.Tensor:
    """Check if the T-shaped object is aligned with the marker.
    
    Args:
        env: The environment instance.
        distance_threshold: Maximum distance for success. Default is 0.02.
        angle_threshold: Maximum orientation error for success. Default is 0.1.
        object_cfg: Configuration for the T-shaped object.
        command_name: Name of the command term.
        
    Returns:
        torch.Tensor: Boolean tensor indicating if alignment is achieved.
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get positions and orientations
    object_pos = object.data.root_pos_w
    command_pos = command[:, 0:3]
    object_quat = object.data.root_quat_w
    command_quat = command[:, 3:7]
    
    # Check position alignment
    distance = torch.norm(object_pos - command_pos, p=2, dim=-1)
    pos_aligned = distance < distance_threshold
    
    # Check orientation alignment
    quat_error = quat_error_magnitude(object_quat, command_quat)
    rot_aligned = quat_error < angle_threshold
    
    return pos_aligned & rot_aligned

def object_away_from_robot(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object")
) -> torch.Tensor:
    """Check if the object has gone too far from the robot.
    
    Args:
        env: The environment instance.
        threshold: Maximum allowed distance between robot and object.
        asset_cfg: Configuration for the robot.
        object_cfg: Configuration for the T-shaped object.
        
    Returns:
        torch.Tensor: Boolean tensor indicating if object is too far.
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Compute distance between robot and object
    dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)
    return dist > threshold 