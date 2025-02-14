from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .tee_command import TeeAlignCommand


@configclass
class TeeAlignCommandCfg(CommandTermCfg):
    """Configuration for the tee alignment command term.

    Please refer to the :class:`TeeAlignCommand` class for more details.
    """

    class_type: type = TeeAlignCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    position_success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reached."""

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal pose when the goal pose is reached."""

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING
    """The configuration for the goal pose visualization marker.""" 