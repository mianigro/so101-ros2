"""Explicit object-in-cup scenario on the reusable SO-101 visual platform."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from so101_rl.paths import CUBE_USD_PATH, CUP_USD_PATH, load_asset_manifest
from so101_rl.tasks.common import (
    SO101_FIXED_JAW_PAD_PRIM_PATH,
    SO101_GRIPPER_CFG,
    SO101_MOVING_JAW_PAD_PRIM_PATH,
    SO101_PAD_THICKNESS_M,
    SO101_ROBOT_JOINT_CFG,
    SO101VisualEnvCfg,
    SO101VisualEventsCfg,
    SO101VisualObservationsCfg,
    SO101VisualSceneCfg,
)
from so101_rl.tasks.common import mdp as common_mdp
from so101_rl.tasks.common.visual_env_cfg import contact_material, contact_properties

from . import mdp

OBJECT_START = (0.20, -0.065, 0.0125)
CUP_START = (0.24, 0.070, 0.0)

# Object centre height while resting on the table (half of the 0.025 m cube).
# Lift pays once the centre clears this baseline by the configured clearance.
OBJECT_REST_HEIGHT = OBJECT_START[2]
OBJECT_LIFT_CLEARANCE = 0.001


@configclass
class ObjectInCupSceneCfg(SO101VisualSceneCfg):
    """Shared workcell plus the cube and cup owned by this scenario."""

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
            collision_props=contact_properties(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            physics_material=contact_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.35, 0.85), roughness=0.48
            ),
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
            collision_props=contact_properties(),
            physics_material=contact_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.35, 0.12), roughness=0.52
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_START),
    )

    fixed_jaw_contact = ContactSensorCfg(
        prim_path=SO101_FIXED_JAW_PAD_PRIM_PATH,
        update_period=0.0,
        history_length=4,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    moving_jaw_contact = ContactSensorCfg(
        prim_path=SO101_MOVING_JAW_PAD_PRIM_PATH,
        update_period=0.0,
        history_length=4,
        track_pose=False,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class ObjectInCupObservationsCfg(SO101VisualObservationsCfg):
    """Deployable actor observations plus task-specific training state."""

    @configclass
    class CriticStateCfg(ObsGroup):
        task_state = ObsTerm(
            func=mdp.critic_task_state,
            params={"robot_cfg": SO101_ROBOT_JOINT_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class ObjectInCupRewardsCfg:
    approach_progress = RewTerm(
        func=mdp.approach_progress,
        weight=1.0,
        params={
            "half_extents": (0.0125, 0.0125, 0.0125),
            "pad_thickness": SO101_PAD_THICKNESS_M,
        },
    )
    closure_progress = RewTerm(
        func=mdp.closure_progress,
        weight=1.0,
        params={
            "half_extents": (0.0125, 0.0125, 0.0125),
            "pad_thickness": SO101_PAD_THICKNESS_M,
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    grasp_acquired = RewTerm(func=mdp.grasp_acquired, weight=2.0)
    lift_progress = RewTerm(
        func=mdp.lift_progress,
        weight=0.1,
        params={
            "lift_clearance": OBJECT_LIFT_CLEARANCE,
            "object_rest_height": OBJECT_REST_HEIGHT,
        },
    )
    transport = RewTerm(
        func=mdp.transport_object,
        weight=3.0,
        params={
            "std": 0.08,
            "minimum_height": OBJECT_REST_HEIGHT + OBJECT_LIFT_CLEARANCE,
        },
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
            "robot_cfg": SO101_GRIPPER_CFG,
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
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    action_rate = RewTerm(func=common_mdp.action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=common_mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SO101_ROBOT_JOINT_CFG},
    )


@configclass
class ObjectInCupTerminationsCfg:
    time_out = DoneTerm(func=common_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.stable_placement,
        params={
            "required_steps": 15,
            "xy_tolerance": 0.0,
            "center_z_min": 0.0,
            "center_z_max": 0.0,
            "linear_velocity_max": 0.025,
            "angular_velocity_max": 0.50,
            "released_position_min": 1.20,
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    dropped = DoneTerm(func=mdp.object_dropped, params={"minimum_height": -0.02})
    invalid = DoneTerm(func=mdp.invalid_state)


@configclass
class ObjectInCupEventsCfg(SO101VisualEventsCfg):
    object_material = EventTerm(
        func=common_mdp.randomize_rigid_body_material,
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
        func=common_mdp.randomize_rigid_body_material,
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
        func=common_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_layout = EventTerm(
        func=mdp.reset_task_layout,
        mode="reset",
        params={
            "curriculum_steps": 45_000_000,
            "object_xy_range_nominal": (0.004, 0.004),
            "object_xy_range_full": (0.025, 0.025),
            "cup_xy_range_nominal": (0.003, 0.003),
            "cup_xy_range_full": (0.020, 0.020),
            "object_yaw_range_full": (-3.141592653589793, 3.141592653589793),
        },
    )
    object_visual = EventTerm(
        func=common_mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": "Object",
            "color_low": (0.04, 0.10, 0.20),
            "color_high": (0.45, 0.75, 1.0),
            "roughness_range": (0.25, 0.85),
        },
    )
    cup_visual = EventTerm(
        func=common_mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": "Cup",
            "color_low": (0.30, 0.08, 0.03),
            "color_high": (1.0, 0.65, 0.35),
            "roughness_range": (0.25, 0.85),
        },
    )


@configclass
class SO101ObjectInCupVisionEnvCfg(SO101VisualEnvCfg):
    scene: ObjectInCupSceneCfg = ObjectInCupSceneCfg(
        # Per-environment USD materials must remain independently authorable.
        num_envs=64,
        env_spacing=0.8,
        replicate_physics=False,
    )
    observations: ObjectInCupObservationsCfg = ObjectInCupObservationsCfg()
    rewards: ObjectInCupRewardsCfg = ObjectInCupRewardsCfg()
    terminations: ObjectInCupTerminationsCfg = ObjectInCupTerminationsCfg()
    events: ObjectInCupEventsCfg = ObjectInCupEventsCfg()

    def __post_init__(self):
        geometry = load_asset_manifest()["geometry"]
        half_extents = tuple(float(value) * 0.5 for value in geometry["cube_extents_m"])
        placement = {
            "xy_tolerance": geometry["success_xy_tolerance_m"],
            "center_z_min": geometry["success_center_z_min_m"],
            "center_z_max": geometry["success_center_z_max_m"],
        }
        self.rewards.insertion.params.update(
            xy_tolerance=placement["xy_tolerance"],
            center_z_max=placement["center_z_max"],
        )
        self.rewards.approach_progress.params["half_extents"] = half_extents
        self.rewards.closure_progress.params["half_extents"] = half_extents
        self.rewards.release.params.update(placement)
        self.rewards.stable.params.update(placement)
        self.terminations.success.params.update(placement)
        super().__post_init__()


@configclass
class SO101ObjectInCupVisionFixedEnvCfg(SO101ObjectInCupVisionEnvCfg):
    """Nominal fixed-pose environment used to prove visual learnability first."""

    def __post_init__(self):
        super().__post_init__()
        self._apply_fixed_mode()
        self.events.object_material = None
        self.events.cup_material = None
        self.events.object_mass = None
        self.events.object_visual = None
        self.events.cup_visual = None
        self.events.reset_layout.params.update(
            object_xy_range_nominal=(0.0, 0.0),
            object_xy_range_full=(0.0, 0.0),
            cup_xy_range_nominal=(0.0, 0.0),
            cup_xy_range_full=(0.0, 0.0),
            object_yaw_range_full=(0.0, 0.0),
        )
