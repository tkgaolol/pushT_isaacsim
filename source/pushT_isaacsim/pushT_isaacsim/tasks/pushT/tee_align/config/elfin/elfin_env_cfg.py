"""Configuration for the Elfin robot arm in tee alignment task."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
import pushT_isaacsim.tasks.pushT.tee_align.tee_align_env_cfg as tee_align_env_cfg

##
# Pre-defined configs
##
import os 
pwd = os.getcwd()

# Robot configuration
ELFIN_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{pwd}/resource/elfin3/elfin3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enable_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "elfin_joint1": 0.0,
            "elfin_joint2": -0.76,
            "elfin_joint3": 1.5,
            "elfin_joint4": 0.0,
            "elfin_joint5": 0.97,
            "elfin_joint6": 0.0,
        },
        pos=(0.0, 0.0, 0.8)
    ),
    actuators={
        "joint_actuator1": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint1"], 
            effort_limit=87.0, 
            velocity_limit=2.175, 
            stiffness=1e7, 
            damping=0
        ),
        "joint_actuator2": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint2"], 
            effort_limit=87.0, 
            velocity_limit=2.175, 
            stiffness=1e7, 
            damping=0
        ),
        "joint_actuator3": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint3"], 
            effort_limit=87.0, 
            velocity_limit=2.175, 
            stiffness=1e7, 
            damping=0
        ),
        "joint_actuator4": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint4"], 
            effort_limit=12.0, 
            velocity_limit=2.61, 
            stiffness=1e7, 
            damping=0
        ),
        "joint_actuator5": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint5"], 
            effort_limit=12.0, 
            velocity_limit=2.61, 
            stiffness=1e7, 
            damping=0
        ),
        "joint_actuator6": ImplicitActuatorCfg(
            joint_names_expr=["elfin_joint6"], 
            effort_limit=12.0, 
            velocity_limit=2.61, 
            stiffness=1e7, 
            damping=0
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Top-down camera
TABLE_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/table_camera",
    eye=(0.5, 0.0, 1.0),  # 1m above table center
    look_at=(0.5, 0.0, 0.0),  # looking at table center
    width=240,
    height=240,
)

@configclass
class ElfinTeeAlignEnvCfg(tee_align_env_cfg.TeeAlignEnvCfg):
    """Configuration for the Elfin robot tee alignment environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set robot configuration
        self.scene.robot = ELFIN_ROBOT_CFG
        # Set camera configuration
        self.scene.table_cam = TABLE_CAM_CFG


@configclass
class ElfinTeeAlignEnvCfg_PLAY(ElfinTeeAlignEnvCfg):
    """Configuration for playing with the environment (smaller number of envs, no randomization)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False 