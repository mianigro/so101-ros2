"""Reusable SO-101 workcell, camera, action, and actor-observation configs."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as UniformNoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from so101_rl.camera_profile import (
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
    focal_length_mm,
    load_camera_profile,
    look_at_opengl_xyzw,
    wxyz_to_xyzw,
)
from so101_rl.paths import (
    CAMERA_SUPPORT_BOTTOM_USD_PATH,
    CAMERA_SUPPORT_TOP_USD_PATH,
    ROBOT_USD_PATH,
    require_vision_assets,
)
from so101_rl.visual_contract import (
    SO101_ARM_ACTION_PATTERN,
    SO101_ARM_DELTA_RAD,
    SO101_ARM_JOINT_NAMES,
    SO101_GRIPPER_DELTA_RAD,
    SO101_JOINT_NAMES,
)

from . import mdp

SO101_ROBOT_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=list(SO101_JOINT_NAMES), preserve_order=True
)
SO101_GRIPPER_CFG = SceneEntityCfg(
    "robot", joint_names=["gripper"], preserve_order=True
)

_PROFILE = load_camera_profile()
_CAMERAS = _PROFILE["cameras"]
_OPTICS = _PROFILE["optics"]
_APERTURE = float(_OPTICS["horizontal_aperture_mm"])
_VERTICAL_APERTURE = _APERTURE * POLICY_IMAGE_HEIGHT / POLICY_IMAGE_WIDTH
_CLIPPING_RANGE = (
    float(_OPTICS["near_clip_m"]),
    float(_OPTICS["far_clip_m"]),
)
_GRIPPER_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link/"
    "lower_arm_link/wrist_link/gripper_link"
)


def contact_properties() -> sim_utils.CollisionPropertiesCfg:
    """Return the contact offsets used by the shared workcell and task props."""
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)


def contact_material() -> sim_utils.RigidBodyMaterialCfg:
    """Return the nominal contact material used by the shared workcell."""
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )


SO101_ROBOT_CFG = ArticulationCfg(
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
            joint_names_expr=list(SO101_ARM_JOINT_NAMES),
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
    joint_ordering=SO101_JOINT_NAMES,
)


def _camera_spawn(name: str) -> sim_utils.PinholeCameraCfg:
    camera = _CAMERAS[name]
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length_mm(
            float(camera["horizontal_fov_deg"]), _APERTURE
        ),
        horizontal_aperture=_APERTURE,
        vertical_aperture=_VERTICAL_APERTURE,
        clipping_range=_CLIPPING_RANGE,
        f_stop=float(_OPTICS["f_stop"]),
        focus_distance=0.4,
        lock_camera=True,
    )


def _quat_mul(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _yaw_quat(degrees: float) -> tuple[float, ...]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _rotate_z(position: tuple[float, ...], degrees: float) -> tuple[float, ...]:
    angle = math.radians(degrees)
    return (
        math.cos(angle) * position[0] - math.sin(angle) * position[1],
        math.sin(angle) * position[0] + math.cos(angle) * position[1],
        position[2],
    )


def _support_part_cfg(camera_name: str, part: str) -> AssetBaseCfg:
    support = _PROFILE["support"]
    scale = float(support["mesh_scale"])
    center = tuple(float(value) for value in support["assembly_center_mm"])
    top_translation = tuple(float(value) for value in support["top_translation_mm"])
    if part == "bottom":
        local_position = (-center[0] * scale, center[2] * scale, -center[1] * scale)
        usd_path = CAMERA_SUPPORT_BOTTOM_USD_PATH
    else:
        relative = tuple(top_translation[i] - center[i] for i in range(3))
        local_position = (
            relative[0] * scale,
            -relative[2] * scale,
            relative[1] * scale,
        )
        usd_path = CAMERA_SUPPORT_TOP_USD_PATH
    yaw = float(support["yaw_deg"][camera_name])
    root = tuple(float(value) for value in support["post_centers_m"][camera_name])
    rotated = _rotate_z(local_position, yaw)
    world_position = tuple(root[i] + rotated[i] for i in range(3))
    upright = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/CameraRig/{camera_name}/{part}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=(scale, scale, scale),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=world_position,
            rot=_quat_mul(_yaw_quat(yaw), upright),
        ),
    )


def _support_collider_cfg(camera_name: str, collider: str) -> AssetBaseCfg:
    support = _PROFILE["support"]
    root = tuple(float(value) for value in support["post_centers_m"][camera_name])
    collision = support["colliders"]
    if collider == "foot":
        size = tuple(float(value) for value in collision["foot_size_m"])
        position = (root[0], root[1], root[2] + size[2] * 0.5)
    else:
        size = tuple(float(value) for value in collision["post_size_m"])
        position = (
            root[0],
            root[1],
            root[2] + float(collision["post_center_z_m"]),
        )
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/CameraRig/{camera_name}/{collider}_collider",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=contact_properties(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


def _housing_cfg(name: str, prim_path: str) -> AssetBaseCfg:
    size = tuple(float(value) for value in _CAMERAS[name]["housing_size_m"])
    return AssetBaseCfg(
        prim_path=f"{prim_path}/Housing",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.015, 0.015, 0.015), roughness=0.55
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, size[2] * 0.5)),
    )


def _nominal_camera_contract() -> dict[str, dict]:
    nominal: dict[str, dict] = {}
    wrist = _CAMERAS["wrist"]
    nominal["wrist"] = {
        "pos": tuple(float(value) for value in wrist["translation_m"]),
        "rot": wxyz_to_xyzw(wrist["orientation_wxyz"]),
        "fov": float(wrist["horizontal_fov_deg"]),
    }
    for name in ("overhead_1", "overhead_2"):
        camera = _CAMERAS[name]
        nominal[name] = {
            "pos": tuple(float(value) for value in camera["position_m"]),
            "rot": look_at_opengl_xyzw(camera["position_m"], camera["look_at_m"]),
            "fov": float(camera["horizontal_fov_deg"]),
        }
    return nominal


SO101_NOMINAL_CAMERAS = _nominal_camera_contract()


@configclass
class SO101VisualSceneCfg(InteractiveSceneCfg):
    """SO-101 workcell and deployment-matched three-camera rig."""

    robot: ArticulationCfg = SO101_ROBOT_CFG

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.46, 0.040),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=contact_properties(),
            physics_material=contact_material(),
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
                prim_path=_GRIPPER_PRIM_PATH,
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
        spawn=sim_utils.DistantLightCfg(
            intensity=1200.0, color=(1.0, 0.95, 0.88)
        ),
    )

    support_1_bottom = _support_part_cfg("overhead_1", "bottom")
    support_1_top = _support_part_cfg("overhead_1", "top")
    support_1_foot = _support_collider_cfg("overhead_1", "foot")
    support_1_post = _support_collider_cfg("overhead_1", "post")
    support_2_bottom = _support_part_cfg("overhead_2", "bottom")
    support_2_top = _support_part_cfg("overhead_2", "top")
    support_2_foot = _support_collider_cfg("overhead_2", "foot")
    support_2_post = _support_collider_cfg("overhead_2", "post")

    # Camera prims must be authored before their housing children. Otherwise USD
    # creates an intermediate Xform that CameraCfg cannot replace with a Camera.
    wrist_camera_prim = AssetBaseCfg(
        prim_path=f"{_GRIPPER_PRIM_PATH}/WristCamera",
        spawn=_camera_spawn("wrist"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SO101_NOMINAL_CAMERAS["wrist"]["pos"],
            rot=SO101_NOMINAL_CAMERAS["wrist"]["rot"],
        ),
    )
    wrist_housing = _housing_cfg("wrist", f"{_GRIPPER_PRIM_PATH}/WristCamera")
    wrist_camera = CameraCfg(
        prim_path=f"{_GRIPPER_PRIM_PATH}/WristCamera",
        update_period=0.05,
        spawn=None,
        data_types=["rgb"],
        width=POLICY_IMAGE_WIDTH,
        height=POLICY_IMAGE_HEIGHT,
    )

    overhead_1_camera_prim = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig/OverheadCamera1",
        spawn=_camera_spawn("overhead_1"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SO101_NOMINAL_CAMERAS["overhead_1"]["pos"],
            rot=SO101_NOMINAL_CAMERAS["overhead_1"]["rot"],
        ),
    )
    overhead_1_housing = _housing_cfg(
        "overhead_1", "{ENV_REGEX_NS}/CameraRig/OverheadCamera1"
    )
    overhead_1_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig/OverheadCamera1",
        update_period=0.05,
        spawn=None,
        data_types=["rgb"],
        width=POLICY_IMAGE_WIDTH,
        height=POLICY_IMAGE_HEIGHT,
    )

    overhead_2_camera_prim = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig/OverheadCamera2",
        spawn=_camera_spawn("overhead_2"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=SO101_NOMINAL_CAMERAS["overhead_2"]["pos"],
            rot=SO101_NOMINAL_CAMERAS["overhead_2"]["rot"],
        ),
    )
    overhead_2_housing = _housing_cfg(
        "overhead_2", "{ENV_REGEX_NS}/CameraRig/OverheadCamera2"
    )
    overhead_2_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig/OverheadCamera2",
        update_period=0.05,
        spawn=None,
        data_types=["rgb"],
        width=POLICY_IMAGE_WIDTH,
        height=POLICY_IMAGE_HEIGHT,
    )


@configclass
class SO101VisualActionsCfg:
    joint_delta = mdp.DelayedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SO101_JOINT_NAMES),
        preserve_order=True,
        scale={
            SO101_ARM_ACTION_PATTERN: SO101_ARM_DELTA_RAD,
            "gripper": SO101_GRIPPER_DELTA_RAD,
        },
        clip={
            SO101_ARM_ACTION_PATTERN: (-SO101_ARM_DELTA_RAD, SO101_ARM_DELTA_RAD),
            "gripper": (-SO101_GRIPPER_DELTA_RAD, SO101_GRIPPER_DELTA_RAD),
        },
        use_zero_offset=True,
        max_delay_steps=1,
    )


def _image_term(sensor_name: str) -> ObsTerm:
    return ObsTerm(
        func=mdp.camera_rgb,
        params={
            "sensor_cfg": SceneEntityCfg(sensor_name),
            "randomize": True,
            "max_delay_steps": 1,
        },
    )


@configclass
class SO101VisualObservationsCfg:
    """The complete and immutable deployable actor observation surface."""

    @configclass
    class JointStateCfg(ObsGroup):
        absolute_joint_positions = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SO101_ROBOT_JOINT_CFG},
            noise=UniformNoise(n_min=-0.005, n_max=0.005),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class WristCfg(ObsGroup):
        rgb = _image_term("wrist_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class Overhead1Cfg(ObsGroup):
        rgb = _image_term("overhead_1_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class Overhead2Cfg(ObsGroup):
        rgb = _image_term("overhead_2_camera")

        def __post_init__(self):
            self.concatenate_terms = True

    joint_state: JointStateCfg = JointStateCfg()
    wrist: WristCfg = WristCfg()
    overhead_1: Overhead1Cfg = Overhead1Cfg()
    overhead_2: Overhead2Cfg = Overhead2Cfg()


@configclass
class SO101VisualEventsCfg:
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
            "asset_cfg": SO101_ROBOT_JOINT_CFG,
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.01, 0.01),
        },
    )
    randomize_cameras = EventTerm(
        func=mdp.randomize_camera_calibration,
        mode="reset",
        params={
            "nominal": SO101_NOMINAL_CAMERAS,
            "width": POLICY_IMAGE_WIDTH,
            "height": POLICY_IMAGE_HEIGHT,
            "overhead_translation_jitter_m": 0.010,
            "wrist_translation_jitter_m": 0.005,
            "rotation_jitter_deg": 3.0,
            "fov_jitter_deg": 3.0,
        },
    )
    table_visual = EventTerm(
        func=mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": "Table",
            "color_low": (0.10, 0.08, 0.06),
            "color_high": (0.60, 0.52, 0.45),
            "roughness_range": (0.35, 0.95),
        },
    )
    lighting = EventTerm(
        func=mdp.randomize_scene_lighting,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        is_global_time=True,
        params={
            "dome_intensity_range": (550.0, 1300.0),
            "distant_intensity_range": (650.0, 1800.0),
            "color_range": (0.78, 1.0),
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
class SO101VisualEnvCfg(ManagerBasedRLEnvCfg):
    """Reusable visual platform; scenarios must supply task rewards and termination."""

    scene: SO101VisualSceneCfg = SO101VisualSceneCfg(
        num_envs=64,
        env_spacing=0.8,
        replicate_physics=False,
    )
    observations: SO101VisualObservationsCfg = SO101VisualObservationsCfg()
    actions: SO101VisualActionsCfg = SO101VisualActionsCfg()
    events: SO101VisualEventsCfg = SO101VisualEventsCfg()
    rewards = None
    terminations = None
    commands = None
    curriculum = None

    def __post_init__(self):
        require_vision_assets()
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

    def _apply_fixed_mode(self) -> None:
        """Disable shared randomization and latency for a scenario's fixed variant."""
        self.actions.joint_delta.max_delay_steps = 0
        self.observations.joint_state.enable_corruption = False
        self.observations.joint_state.absolute_joint_positions.noise = None
        for group_name in ("wrist", "overhead_1", "overhead_2"):
            term = getattr(self.observations, group_name).rgb
            term.params["randomize"] = False
            term.params["max_delay_steps"] = 0

        self.events.actuator_response = None
        self.events.randomize_cameras = None
        self.events.table_visual = None
        self.events.lighting = None
        self.events.reset_robot.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot.params["velocity_range"] = (0.0, 0.0)

    def play_mode(self):
        super().play_mode()
        self.scene.ee_frame.debug_vis = True
        self.observations.joint_state.enable_corruption = False


__all__ = [
    "SO101_GRIPPER_CFG",
    "SO101_NOMINAL_CAMERAS",
    "SO101PhysicsCfg",
    "SO101_ROBOT_CFG",
    "SO101_ROBOT_JOINT_CFG",
    "SO101VisualActionsCfg",
    "SO101VisualEnvCfg",
    "SO101VisualEventsCfg",
    "SO101VisualObservationsCfg",
    "SO101VisualSceneCfg",
    "contact_material",
    "contact_properties",
]
