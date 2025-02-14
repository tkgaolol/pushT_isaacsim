import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.assets import RigidObject

def tee_distance_reward(
    env: ManagerBasedRLEnv, 
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    marker_cfg: SceneEntityCfg = SceneEntityCfg("tee_marker")
) -> torch.Tensor:
    """Compute the distance reward between the T-shaped object and marker using tanh kernel.
    
    Args:
        env: The environment instance.
        std: Standard deviation for the tanh kernel. Default is 0.1.
        object_cfg: Configuration for the T-shaped object.
        marker_cfg: Configuration for the target marker.
        
    Returns:
        torch.Tensor: Distance-based reward scaled between 0 and 1.
    """
    object: RigidObject = env.scene[object_cfg.name]
    marker: RigidObject = env.scene[marker_cfg.name]
    
    object_pos = object.data.root_pos_w
    marker_pos = marker.data.root_pos_w
    distance = torch.norm(object_pos - marker_pos, p=2, dim=-1)
    return 1.0 - torch.tanh(distance / std)

def tee_orientation_reward(
    env: ManagerBasedRLEnv, 
    rot_eps: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tee_object"),
    marker_cfg: SceneEntityCfg = SceneEntityCfg("tee_marker")
) -> torch.Tensor:
    """Compute the orientation alignment reward between the T-shaped object and marker.
    
    Args:
        env: The environment instance.
        rot_eps: Small constant to prevent division by zero. Default is 0.1.
        object_cfg: Configuration for the T-shaped object.
        marker_cfg: Configuration for the target marker.
        
    Returns:
        torch.Tensor: Orientation-based reward using inverse of quaternion error.
    """
    object: RigidObject = env.scene[object_cfg.name]
    marker: RigidObject = env.scene[marker_cfg.name]
    
    object_quat = object.data.root_quat_w
    marker_quat = marker.data.root_quat_w
    quat_error = quat_error_magnitude(object_quat, marker_quat)
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
    marker_cfg: SceneEntityCfg = SceneEntityCfg("tee_marker")
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
    marker: RigidObject = env.scene[marker_cfg.name]
    
    object_pos = object.data.root_pos_w
    marker_pos = marker.data.root_pos_w
    object_quat = object.data.root_quat_w
    marker_quat = marker.data.root_quat_w
    
    distance = torch.norm(object_pos - marker_pos, p=2, dim=-1)
    quat_error = quat_error_magnitude(object_quat, marker_quat)
    
    is_aligned = (distance < distance_threshold) & (quat_error < angle_threshold)
    return is_aligned.float() 