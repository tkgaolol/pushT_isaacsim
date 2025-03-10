from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import os
pwd = os.getcwd()
##
# Scene definition
##
@configclass
class TeeAlignSceneCfg(InteractiveSceneCfg):
    """Configuration for the tee alignment scene with an Elfin robot arm."""

    # Robot configuration
    robot: ArticulationCfg = MISSING

    # # Top-down camera
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.275, -0.05, 2), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    #     data_types=["rgb", "depth", "semantic_segmentation"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=640,
    #     height=480,
    # )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0), rot=(0, 0, 0, 0)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    # T-shaped object to be pushed
    tee_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TeeObject",
        spawn=UsdFileCfg(
            scale=(0.0005, 0.0005, 0.0005),
            usd_path=f"{pwd}/resource/Tee.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0, 0),  # On table surface
            # rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
            # lin_vel=(0.0, 0.0, 0.0),  # No initial velocity
            # ang_vel=(0.0, 0.0, 0.0),  # No initial angular velocity
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    tee_pose = mdp.TeeAlignCommandCfg(
        asset_name="tee_object",
        update_goal_on_success=True,
        position_success_threshold=0.02,  # 2cm
        orientation_success_threshold=0.1,  # ~5.7 degrees
        make_quat_unique=False,
        marker_pos_offset=(0.0, 0.0, 0.0), 
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_marker",
            markers={
                "goal": UsdFileCfg(
                    usd_path=f"{pwd}/resource/Tee.usd",
                    scale=(0.0005, 0.0005, 0.0005),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            },
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        tee_object_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("tee_object")}
        )
        tee_object_rot = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("tee_object"), "make_quat_unique": False}
        )
        # tee_marker_pos = ObsTerm(func=mdp.tee_marker_pos)
        # tee_marker_rot = ObsTerm(func=mdp.tee_marker_rot)
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "tee_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("tee_object"), "command_name": "tee_pose", "make_quat_unique": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # @configclass
    # class RGBCameraPolicyCfg(ObsGroup):
    #     """Observations for policy group with RGB images."""

    #     image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})
    #     # image = ObsTerm(
    #     #     func=mdp.image_features,
    #     #     params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
    #     # )
    #     # image = ObsTerm(
    #     #     func=mdp.image_features,
    #     #     params={
    #     #         "sensor_cfg": SceneEntityCfg("tiled_camera"),
    #     #         "data_type": "rgb",
    #     #         "model_name": "theia-tiny-patch16-224-cddsv",
    #     #         "model_device": "cuda:0",
    #     #     },
    #     # )

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    # rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.2, 0.4), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "pose_range": {"x": (0.3, 0.3), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tee_object"),
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.set_default_joint_pose,
    #     mode="reset",
    #     params={
    #         "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance between tee object and marker
    tee_distance = RewTerm(
        func=mdp.tee_distance_reward, 
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("tee_object"), "command_name": "tee_pose"}
    )
    # Orientation alignment between tee object and marker
    tee_orientation = RewTerm(
        func=mdp.tee_orientation_reward, 
        weight=0.5,
        params={"object_cfg": SceneEntityCfg("tee_object"), "command_name": "tee_pose", "rot_eps": 0.1}
    )
    # Success bonus
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("tee_object"), "command_name": "tee_pose"}
    )
    # Penalty for excessive movement
    # action_penalty = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.1)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.tee_aligned)
    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_robot, 
        params={"threshold": 0.95, "asset_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("tee_object")}
    )

@configclass
class TeeAlignEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the tee alignment environment."""

    # Scene settings
    scene: TeeAlignSceneCfg = TeeAlignSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 5
        self.episode_length_s = 20.0  # 20 seconds per episode
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Viewer settings
        self.viewer.eye = (1.0, 1.0, 1.0) 