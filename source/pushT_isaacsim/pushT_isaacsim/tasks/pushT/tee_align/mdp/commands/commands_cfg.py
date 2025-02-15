from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .tee_command import TeeAlignCommand

import os
pwd = os.getcwd()

@configclass
class TeeAlignCommandCfg(CommandTermCfg):
    """Configuration for the tee alignment command term.

    Please refer to the :class:`TeeAlignCommand` class for more details.
    """

    class_type: type = TeeAlignCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = "tee_object"
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position."""

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not."""

    position_success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reached."""

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal pose when the goal pose is reached."""

    marker_pos_offset: tuple[float, float, float] = MISSING
    """Position offset of the marker from the object's desired position."""

    debug_vis: bool = MISSING
    """Whether to enable debug visualization."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{pwd}/resource/Tee.usd",
                scale=(0.0005, 0.0005, 0.0005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        },
    )
    """The configuration for the goal pose visualization marker.""" 