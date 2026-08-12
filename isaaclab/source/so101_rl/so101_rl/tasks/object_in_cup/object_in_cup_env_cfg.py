"""Shared simulation configuration for SO-101 visual reinforcement learning."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from so101_rl.paths import (
    CUBE_USD_PATH,
    CUP_USD_PATH,
    ROBOT_USD_PATH,
    load_asset_manifest,
)

from . import mdp

SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_ARM_JOINTS = SO101_JOINTS[:-1]

OBJECT_START = (0.20, -0.065, 0.0125)
CUP_START = (0.24, 0.070, 0.0)

SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path="/Geometry",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The base meshes extend 2.4 mm below base_link in this repository's URDF.
        pos=(0.0, 0.0, 0.0024),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.60,
            "elbow_flex": 0.80,
            "wrist_flex": 0.60,
            "wrist_roll": 0.0,
            "gripper": 1.50,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(SO101_ARM_JOINTS),
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=0.98,
    joint_ordering=SO101_JOINTS,
)


def _contact_properties() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)


def _contact_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )


@configclass
class ObjectInCupSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_CFG

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CUBE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=_contact_properties(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            physics_material=_contact_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START),
    )

    cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CUP_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=_contact_properties(),
            physics_material=_contact_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_START),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.46, 0.040),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=_contact_properties(),
            physics_material=_contact_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.24, 0.20)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.14, 0.0, -0.020)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.12, 0.12, 0.12)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.041)),
        collision_group=-1,
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link/"
                    "lower_arm_link/wrist_link/gripper_link"
                ),
                name="grasp_frame",
                offset=OffsetCfg(
                    pos=(-0.0079, -0.000218121, -0.0981274),
                    rot=(0.0, 1.0, 0.0, 0.0),
                ),
            )
        ],
        debug_vis=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.90, 0.90, 0.90)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.95, 0.88)),
    )


_ROBOT_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=list(SO101_JOINTS), preserve_order=True
)
_GRIPPER_CFG = SceneEntityCfg("robot", joint_names=["gripper"], preserve_order=True)


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach_object, weight=1.0, params={"std": 0.06})
    grasp = RewTerm(
        func=mdp.grasp_object,
        weight=0.5,
        params={
            "distance_threshold": 0.035,
            "closed_position_max": 0.45,
            "robot_cfg": _GRIPPER_CFG,
        },
    )
    lift = RewTerm(func=mdp.lift_object, weight=2.0, params={"lift_height": 0.075})
    transport = RewTerm(
        func=mdp.transport_object,
        weight=3.0,
        params={"std": 0.08, "minimum_height": 0.045},
    )
    insertion = RewTerm(
        func=mdp.insert_object,
        weight=5.0,
        params={"xy_tolerance": 0.0, "center_z_max": 0.0, "approach_height": 0.090},
    )
    release = RewTerm(
        func=mdp.release_object,
        weight=8.0,
        params={
            "xy_tolerance": 0.0,
            "center_z_min": 0.0,
            "center_z_max": 0.0,
            "released_position_min": 1.20,
            "robot_cfg": _GRIPPER_CFG,
        },
    )
    stable = RewTerm(
        func=mdp.stable_placement_reward,
        weight=20.0,
        params={
            "xy_tolerance": 0.0,
            "center_z_min": 0.0,
            "center_z_max": 0.0,
            "linear_velocity_max": 0.025,
            "angular_velocity_max": 0.50,
            "released_position_min": 1.20,
            "robot_cfg": _GRIPPER_CFG,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": _ROBOT_JOINT_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.stable_placement,
        params={
            "required_steps": 10,
            "xy_tolerance": 0.0,
            "center_z_min": 0.0,
            "center_z_max": 0.0,
            "linear_velocity_max": 0.025,
            "angular_velocity_max": 0.50,
            "released_position_min": 1.20,
            "robot_cfg": _GRIPPER_CFG,
        },
    )
    dropped = DoneTerm(func=mdp.object_dropped, params={"minimum_height": -0.02})
    invalid = DoneTerm(func=mdp.invalid_state)


@configclass
class EventsCfg:
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.55, 1.05),
            "dynamic_friction_range": (0.40, 0.85),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    cup_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "static_friction_range": (0.55, 1.05),
            "dynamic_friction_range": (0.40, 0.85),
            "restitution_range": (0.0, 0.03),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    actuator_response = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": _ROBOT_JOINT_CFG,
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.01, 0.01),
        },
    )
    reset_layout = EventTerm(
        func=mdp.reset_task_layout,
        mode="reset",
        params={
            "curriculum_steps": 30_000_000,
            "object_xy_range_nominal": (0.004, 0.004),
            "object_xy_range_full": (0.025, 0.025),
            "cup_xy_range_nominal": (0.003, 0.003),
            "cup_xy_range_full": (0.020, 0.020),
            "object_yaw_range_full": (-3.141592653589793, 3.141592653589793),
        },
    )


@configclass
class SO101PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.01,
        friction_correlation_distance=0.00625,
        solve_articulation_contact_last=True,
        gpu_max_rigid_patch_count=5 * 2**15,
        gpu_found_lost_pairs_capacity=2**25,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class SO101ObjectInCupBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Common scene and MDP configuration completed by a visual task subclass."""

    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        manifest = load_asset_manifest()
        geometry = manifest["geometry"]
        placement = {
            "xy_tolerance": geometry["success_xy_tolerance_m"],
            "center_z_min": geometry["success_center_z_min_m"],
            "center_z_max": geometry["success_center_z_max_m"],
        }
        self.rewards.insertion.params.update(
            xy_tolerance=placement["xy_tolerance"],
            center_z_max=placement["center_z_max"],
        )
        self.rewards.release.params.update(placement)
        self.rewards.stable.params.update(placement)
        self.terminations.success.params.update(placement)

        self.decimation = 5
        self.episode_length_s = 15.0
        self.is_finite_horizon = False
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = SO101PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(0.48, -0.48, 0.34),
            lookat=(0.17, 0.0, 0.09),
        )

    def play_mode(self):
        super().play_mode()
        self.scene.ee_frame.debug_vis = True
