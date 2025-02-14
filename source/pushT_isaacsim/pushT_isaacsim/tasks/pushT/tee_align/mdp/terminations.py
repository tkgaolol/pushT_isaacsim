import torch

from isaaclab.envs.mdp.terminations import check_timeout
from isaaclab.utils.torch_utils import quat_diff_rad

def time_out(env, _) -> torch.Tensor:
    """Check if the episode has timed out."""
    return check_timeout(env)

def tee_aligned(env, _) -> torch.Tensor:
    """Check if the T-shaped object is aligned with the marker."""
    # Get positions
    object_pos = env.scene.get_frame_world_position("tee_object")
    marker_pos = env.scene.get_frame_world_position("tee_marker")
    
    # Get orientations
    object_quat = env.scene.get_frame_world_quaternion("tee_object")
    marker_quat = env.scene.get_frame_world_quaternion("tee_marker")
    
    # Check position alignment
    pos_diff = torch.norm(object_pos - marker_pos, dim=-1)
    pos_aligned = pos_diff < 0.02  # 2cm threshold
    
    # Check orientation alignment
    angle_diff = quat_diff_rad(object_quat, marker_quat)
    rot_aligned = angle_diff < 0.1  # ~5.7 degrees threshold
    
    return pos_aligned & rot_aligned 