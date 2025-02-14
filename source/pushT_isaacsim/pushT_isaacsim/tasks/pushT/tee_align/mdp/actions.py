from dataclasses import dataclass, field
from typing import List
import math
from isaaclab.envs.mdp.actions import JointPositionActionBaseCfg

@dataclass
class JointPositionActionCfg(JointPositionActionBaseCfg):
    """Configuration for joint position actions."""

    # Default joint limits for Elfin robot
    joint_limits: List[List[float]] = field(
        default_factory=lambda: [
            [-179.909 * math.pi / 180.0, 179.909 * math.pi / 180.0],  # joint 1
            [-134.645 * math.pi / 180.0, 134.645 * math.pi / 180.0],  # joint 2
            [-149.542 * math.pi / 180.0, 149.542 * math.pi / 180.0],  # joint 3
            [-179.909 * math.pi / 180.0, 179.909 * math.pi / 180.0],  # joint 4
            [-146.677 * math.pi / 180.0, 146.677 * math.pi / 180.0],  # joint 5
            [-179.909 * math.pi / 180.0, 179.909 * math.pi / 180.0],  # joint 6
        ]
    )
    
    # Maximum joint velocity limits
    joint_velocity_limits: List[float] = field(
        default_factory=lambda: [3.14159] * 6  # 180 deg/s for all joints
    ) 