"""Sub-module containing command generators for tee alignment task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import TeeAlignCommandCfg


class TeeAlignCommand(CommandTerm):
    """Command term that generates pose commands for tee alignment task.

    This command term generates pose commands for the T-shaped object. The commands include both
    position and orientation targets. The position targets are sampled within a reasonable range
    on the table surface, while orientation targets are sampled to create meaningful alignment
    challenges.

    Unlike typical command terms that resample based on time, this command term resamples the
    goals when the object reaches the current goal pose. The goal pose is considered reached
    when both position and orientation errors are below their respective thresholds.
    """

    cfg: TeeAlignCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TeeAlignCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- unit vectors for rotation sampling
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TeeAlignCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in world frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_w, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # -- compute the number of consecutive successes
        successes = (
            self.metrics["position_error"] < self.cfg.position_success_threshold
        ) & (self.metrics["orientation_error"] < self.cfg.orientation_success_threshold)
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets on table surface
        # -- x: forward/back on table
        pos_x = 0.5 + 0.2 * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)  # 0.3 to 0.7
        # -- y: left/right on table
        pos_y = 0.2 * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)  # -0.2 to 0.2
        # -- z: fixed height above table
        pos_z = torch.full((len(env_ids),), 0.05, device=self.device)
        # combine positions
        self.pos_command_w[env_ids] = torch.stack((pos_x, pos_y, pos_z), dim=-1)

        # sample new orientation targets
        # -- only rotate around z-axis for 2D alignment
        rand_angle = 2.0 * torch.pi * torch.rand(len(env_ids), device=self.device)  # 0 to 2π
        quat = math_utils.quat_from_angle_axis(rand_angle, self._Z_UNIT_VEC[env_ids])
        # make quaternion unique if configured
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = (
                self.metrics["position_error"] < self.cfg.position_success_threshold
            ) & (self.metrics["orientation_error"] < self.cfg.orientation_success_threshold)
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat) 