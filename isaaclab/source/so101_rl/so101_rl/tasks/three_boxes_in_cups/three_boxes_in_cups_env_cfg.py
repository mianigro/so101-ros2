"""Three interchangeable boxes and cups on the reusable SO-101 platform."""

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

BOX_STARTS = (
    (0.15, -0.085, 0.0125),
    (0.22, -0.120, 0.0125),
    (0.29, -0.085, 0.0125),
)
CUP_STARTS = (
    (0.15, 0.085, 0.0),
    (0.22, 0.120, 0.0),
    (0.29, 0.085, 0.0),
)
BOX_NAMES = ("box_1", "box_2", "box_3")
CUP_NAMES = ("cup_1", "cup_2", "cup_3")

# Box centre height while resting on the table (half of the 0.025 m cube).
# Lift pays once the centre clears this baseline by the configured clearance.
BOX_REST_HEIGHT = BOX_STARTS[0][2]
BOX_LIFT_CLEARANCE = 0.001

PLACEMENT_PARAMS = {
    "xy_tolerance": 0.0,
    "center_z_min": 0.0,
    "center_z_max": 0.0,
    "linear_velocity_max": 0.025,
    "angular_velocity_max": 0.50,
    "released_position_min": 1.20,
    "release_distance_min": 0.050,
}


def _box_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Box{index + 1}",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_STARTS[index]),
    )


def _cup_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Cup{index + 1}",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_STARTS[index]),
    )


@configclass
class ThreeBoxesInCupsSceneCfg(SO101VisualSceneCfg):
    box_1: RigidObjectCfg = _box_cfg(0)
    box_2: RigidObjectCfg = _box_cfg(1)
    box_3: RigidObjectCfg = _box_cfg(2)
    cup_1: RigidObjectCfg = _cup_cfg(0)
    cup_2: RigidObjectCfg = _cup_cfg(1)
    cup_3: RigidObjectCfg = _cup_cfg(2)
    fixed_jaw_contact = ContactSensorCfg(
        prim_path=SO101_FIXED_JAW_PAD_PRIM_PATH,
        update_period=0.0,
        history_length=4,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Box1",
            "{ENV_REGEX_NS}/Box2",
            "{ENV_REGEX_NS}/Box3",
        ],
    )
    moving_jaw_contact = ContactSensorCfg(
        prim_path=SO101_MOVING_JAW_PAD_PRIM_PATH,
        update_period=0.0,
        history_length=4,
        track_pose=False,
        force_threshold=0.1,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Box1",
            "{ENV_REGEX_NS}/Box2",
            "{ENV_REGEX_NS}/Box3",
        ],
    )


@configclass
class ThreeBoxesInCupsObservationsCfg(SO101VisualObservationsCfg):
    @configclass
    class CriticStateCfg(ObsGroup):
        task_state = ObsTerm(
            func=mdp.critic_task_state,
            params={
                "robot_cfg": SO101_ROBOT_JOINT_CFG,
                "box_names": BOX_NAMES,
                "cup_names": CUP_NAMES,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic_state: CriticStateCfg = CriticStateCfg()


@configclass
class ThreeBoxesInCupsRewardsCfg:
    approach_progress = RewTerm(
        func=mdp.approach_progress,
        weight=1.0,
        params={
            "half_extents": (0.0125, 0.0125, 0.0125),
            "pad_thickness": SO101_PAD_THICKNESS_M,
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    closure_progress = RewTerm(
        func=mdp.closure_progress,
        weight=1.0,
        params={
            "half_extents": (0.0125, 0.0125, 0.0125),
            "pad_thickness": SO101_PAD_THICKNESS_M,
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    grasp_acquired = RewTerm(
        func=mdp.grasp_acquired,
        weight=2.0,
        params={
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    lift_progress = RewTerm(
        func=mdp.lift_progress,
        weight=0.1,
        params={
            "lift_clearance": BOX_LIFT_CLEARANCE,
            "object_rest_height": BOX_REST_HEIGHT,
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    transport = RewTerm(
        func=mdp.transport_to_empty_cup,
        weight=3.0,
        params={
            "std": 0.08,
            "minimum_height": BOX_REST_HEIGHT + BOX_LIFT_CLEARANCE,
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    insertion = RewTerm(
        func=mdp.insertion_progress,
        weight=5.0,
        params={"xy_tolerance": 0.0, "center_z_max": 0.0, "approach_height": 0.090},
    )
    release = RewTerm(
        func=mdp.released_placement_progress,
        weight=8.0,
        params={
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    stable = RewTerm(
        func=mdp.stable_placement_progress,
        weight=20.0,
        params={
            "placement": dict(PLACEMENT_PARAMS),
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
class ThreeBoxesInCupsTerminationsCfg:
    time_out = DoneTerm(func=common_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.all_boxes_stably_placed,
        params={
            "required_steps": 15,
            "placement": dict(PLACEMENT_PARAMS),
            "robot_cfg": SO101_GRIPPER_CFG,
        },
    )
    dropped = DoneTerm(func=mdp.any_box_dropped, params={"minimum_height": -0.02})
    invalid = DoneTerm(func=mdp.invalid_state)


def _material_event(asset_name: str, *, cup: bool = False) -> EventTerm:
    return EventTerm(
        func=common_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "static_friction_range": (0.55, 1.05),
            "dynamic_friction_range": (0.40, 0.85),
            "restitution_range": (0.0, 0.03 if cup else 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )


def _mass_event(asset_name: str) -> EventTerm:
    return EventTerm(
        func=common_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )


def _visual_event(asset_name: str, *, cup: bool = False) -> EventTerm:
    return EventTerm(
        func=common_mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": asset_name,
            "color_low": (0.30, 0.08, 0.03) if cup else (0.04, 0.10, 0.20),
            "color_high": (1.0, 0.65, 0.35) if cup else (0.45, 0.75, 1.0),
            "roughness_range": (0.25, 0.85),
        },
    )


@configclass
class ThreeBoxesInCupsEventsCfg(SO101VisualEventsCfg):
    box_1_material = _material_event("box_1")
    box_2_material = _material_event("box_2")
    box_3_material = _material_event("box_3")
    cup_1_material = _material_event("cup_1", cup=True)
    cup_2_material = _material_event("cup_2", cup=True)
    cup_3_material = _material_event("cup_3", cup=True)
    box_1_mass = _mass_event("box_1")
    box_2_mass = _mass_event("box_2")
    box_3_mass = _mass_event("box_3")
    reset_layout = EventTerm(
        func=mdp.reset_three_box_layout,
        mode="reset",
        params={
            "curriculum_steps": 45_000_000,
            "nominal_jitter": 0.003,
            "box_zone_low": (0.14, -0.13),
            "box_zone_high": (0.30, -0.04),
            "cup_zone_low": (0.14, 0.04),
            "cup_zone_high": (0.30, 0.13),
            "box_minimum_separation": 0.040,
            "cup_minimum_separation": 0.060,
            "cross_minimum_separation": 0.055,
            "max_attempts": 128,
            "fixed_layout": False,
            "box_names": BOX_NAMES,
            "cup_names": CUP_NAMES,
        },
    )
    box_1_visual = _visual_event("Box1")
    box_2_visual = _visual_event("Box2")
    box_3_visual = _visual_event("Box3")
    cup_1_visual = _visual_event("Cup1", cup=True)
    cup_2_visual = _visual_event("Cup2", cup=True)
    cup_3_visual = _visual_event("Cup3", cup=True)


@configclass
class SO101ThreeBoxesInCupsVisionEnvCfg(SO101VisualEnvCfg):
    scene: ThreeBoxesInCupsSceneCfg = ThreeBoxesInCupsSceneCfg(
        num_envs=64,
        env_spacing=0.8,
        replicate_physics=False,
    )
    observations: ThreeBoxesInCupsObservationsCfg = (
        ThreeBoxesInCupsObservationsCfg()
    )
    rewards: ThreeBoxesInCupsRewardsCfg = ThreeBoxesInCupsRewardsCfg()
    terminations: ThreeBoxesInCupsTerminationsCfg = ThreeBoxesInCupsTerminationsCfg()
    events: ThreeBoxesInCupsEventsCfg = ThreeBoxesInCupsEventsCfg()

    def __post_init__(self):
        geometry = load_asset_manifest()["geometry"]
        half_extents = tuple(float(value) * 0.5 for value in geometry["cube_extents_m"])
        placement = {
            "xy_tolerance": geometry["success_xy_tolerance_m"],
            "center_z_min": geometry["success_center_z_min_m"],
            "center_z_max": geometry["success_center_z_max_m"],
        }
        for term_name in (
            "approach_progress",
            "closure_progress",
            "grasp_acquired",
            "lift_progress",
            "transport",
            "release",
            "stable",
        ):
            getattr(self.rewards, term_name).params["placement"].update(placement)
        self.rewards.approach_progress.params["half_extents"] = half_extents
        self.rewards.closure_progress.params["half_extents"] = half_extents
        self.rewards.insertion.params.update(
            xy_tolerance=placement["xy_tolerance"],
            center_z_max=placement["center_z_max"],
        )
        self.terminations.success.params["placement"].update(placement)
        super().__post_init__()
        self.episode_length_s = 45.0


@configclass
class SO101ThreeBoxesInCupsVisionFixedEnvCfg(
    SO101ThreeBoxesInCupsVisionEnvCfg
):
    """Deterministic three-box layout used before randomized training."""

    def __post_init__(self):
        super().__post_init__()
        self._apply_fixed_mode()
        for prefix in (*BOX_NAMES, *CUP_NAMES):
            setattr(self.events, f"{prefix}_material", None)
            setattr(self.events, f"{prefix}_visual", None)
        for name in BOX_NAMES:
            setattr(self.events, f"{name}_mass", None)
        self.events.reset_layout.params["fixed_layout"] = True
