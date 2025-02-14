import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.assets import RigidObject

def tee_distance_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    command_name: str = "tee_pose"
) -> torch.Tensor:
    """Compute the distance reward between the T-shaped object and marker using tanh kernel.
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the T-shaped object.
        command_name: Name of the command term.
        
    Returns:
        torch.Tensor: Distance-based reward scaled between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    object_pos = object.data.root_pos_w - env.scene.env_origins
    command_pos = command[:, 0:3]
    distance = torch.norm(object_pos - command_pos, p=2, dim=-1)
    return distance

def tee_orientation_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    command_name: str = "tee_pose",
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Compute the orientation alignment reward between the T-shaped object and marker.
    
    Args:
        env: The environment instance.
        rot_eps: Small constant to prevent division by zero. Default is 0.1.
        object_cfg: Configuration for the T-shaped object.
        command_name: Name of the command term.
                
    Returns:
        torch.Tensor: Orientation-based reward using inverse of quaternion error.
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    object_quat = object.data.root_quat_w
    command_quat = command[:, 3:7]
    quat_error = quat_error_magnitude(object_quat, command_quat)
    return 1.0 / (quat_error + rot_eps)

def action_magnitude_penalty(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    """Compute penalty for large action magnitudes.
    
    Args:
        env: The environment instance.
        scale: Scaling factor for the penalty. Default is 0.1.
        
    Returns:
        torch.Tensor: Negative reward proportional to squared action magnitude.
    """
    return -scale * torch.sum(env.action_manager.action ** 2, dim=-1)

def success_bonus(
    env: ManagerBasedRLEnv, 
    distance_threshold: float = 0.02, 
    angle_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    command_name: str = "tee_pose"
) -> torch.Tensor:
    """Provide a bonus reward when the object is successfully aligned with the marker.
    
    Args:
        env: The environment instance.
        distance_threshold: Maximum distance for success. Default is 0.02.
        angle_threshold: Maximum orientation error for success. Default is 0.1.
        object_cfg: Configuration for the T-shaped object.
        marker_cfg: Configuration for the target marker.
        
    Returns:
        torch.Tensor: Binary reward (1.0 for success, 0.0 otherwise).
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    object_pos = object.data.root_pos_w - env.scene.env_origins
    command_pos = command[:, 0:3]
    object_quat = object.data.root_quat_w
    command_quat = command[:, 3:7]
    
    distance = torch.norm(object_pos - command_pos, p=2, dim=-1)
    quat_error = quat_error_magnitude(object_quat, command_quat)
    
    is_aligned = (distance < distance_threshold) & (quat_error < angle_threshold)
    return is_aligned.float() 