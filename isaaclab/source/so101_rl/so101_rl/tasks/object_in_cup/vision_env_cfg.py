"""Three-camera deployable SO-101 object-in-cup environments."""

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as UniformNoise

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
    require_vision_assets,
)

from . import mdp
from .object_in_cup_env_cfg import (
    SO101_JOINTS,
    ActionsCfg,
    EventsCfg,
    ObjectInCupSceneCfg,
    RewardsCfg,
    SO101ObjectInCupEnvCfg,
    TerminationsCfg,
    _GRIPPER_CFG,
    _ROBOT_JOINT_CFG,
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
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.0
            ),
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


def _vision_object_cfgs():
    nominal = ObjectInCupSceneCfg(num_envs=1, env_spacing=0.8)
    object_cfg = copy.deepcopy(nominal.object)
    object_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.12, 0.35, 0.85), roughness=0.48
    )
    cup_cfg = copy.deepcopy(nominal.cup)
    cup_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.85, 0.35, 0.12), roughness=0.52
    )
    return object_cfg, cup_cfg


_VISION_OBJECT_CFG, _VISION_CUP_CFG = _vision_object_cfgs()


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


NOMINAL_CAMERAS = _nominal_camera_contract()


@configclass
class ObjectInCupVisionSceneCfg(ObjectInCupSceneCfg):
    """State scene plus the YAML-derived three-camera rig in every environment."""

    object = _VISION_OBJECT_CFG
    cup = _VISION_CUP_CFG

    support_1_bottom = _support_part_cfg("overhead_1", "bottom")
    support_1_top = _support_part_cfg("overhead_1", "top")
    support_1_foot = _support_collider_cfg("overhead_1", "foot")
    support_1_post = _support_collider_cfg("overhead_1", "post")
    support_2_bottom = _support_part_cfg("overhead_2", "bottom")
    support_2_top = _support_part_cfg("overhead_2", "top")
    support_2_foot = _support_collider_cfg("overhead_2", "foot")
    support_2_post = _support_collider_cfg("overhead_2", "post")

    # Author camera prims as ordinary scene assets before their housing children.
    # If a housing child is spawned first, USD creates an intermediate Xform at
    # the camera path and CameraCfg will not replace it with a Camera prim.
    wrist_camera_prim = AssetBaseCfg(
        prim_path=f"{_GRIPPER_PRIM_PATH}/WristCamera",
        spawn=_camera_spawn("wrist"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=NOMINAL_CAMERAS["wrist"]["pos"],
            rot=NOMINAL_CAMERAS["wrist"]["rot"],
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
            pos=NOMINAL_CAMERAS["overhead_1"]["pos"],
            rot=NOMINAL_CAMERAS["overhead_1"]["rot"],
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
            pos=NOMINAL_CAMERAS["overhead_2"]["pos"],
            rot=NOMINAL_CAMERAS["overhead_2"]["rot"],
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
class VisionActionsCfg(ActionsCfg):
    joint_delta = mdp.DelayedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SO101_JOINTS),
        preserve_order=True,
        scale={"shoulder_.*|elbow_flex|wrist_.*": 0.05, "gripper": 0.15},
        clip={
            "shoulder_.*|elbow_flex|wrist_.*": (-0.05, 0.05),
            "gripper": (-0.15, 0.15),
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
class VisionObservationsCfg:
    @configclass
    class JointStateCfg(ObsGroup):
        absolute_joint_positions = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _ROBOT_JOINT_CFG},
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

    @configclass
    class CriticCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ROBOT_JOINT_CFG})
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ROBOT_JOINT_CFG})
        previous_action = ObsTerm(func=mdp.last_action)
        gripper_object_delta = ObsTerm(func=mdp.ee_to_object)
        object_cup_delta = ObsTerm(func=mdp.object_to_cup)
        object_quaternion = ObsTerm(func=mdp.object_orientation)
        object_velocity = ObsTerm(func=mdp.object_velocity)
        gripper_position = ObsTerm(
            func=mdp.gripper_position, params={"robot_cfg": _GRIPPER_CFG}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    joint_state: JointStateCfg = JointStateCfg()
    wrist: WristCfg = WristCfg()
    overhead_1: Overhead1Cfg = Overhead1Cfg()
    overhead_2: Overhead2Cfg = Overhead2Cfg()
    critic: CriticCfg = CriticCfg()


@configclass
class VisionEventsCfg(EventsCfg):
    randomize_cameras = EventTerm(
        func=mdp.randomize_camera_calibration,
        mode="reset",
        params={
            "nominal": NOMINAL_CAMERAS,
            "width": POLICY_IMAGE_WIDTH,
            "height": POLICY_IMAGE_HEIGHT,
            "overhead_translation_jitter_m": 0.010,
            "wrist_translation_jitter_m": 0.005,
            "rotation_jitter_deg": 3.0,
            "fov_jitter_deg": 3.0,
        },
    )
    object_visual = EventTerm(
        func=mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": "Object",
            "color_low": (0.04, 0.10, 0.20),
            "color_high": (0.45, 0.75, 1.0),
            "roughness_range": (0.25, 0.85),
        },
    )
    cup_visual = EventTerm(
        func=mdp.randomize_preview_material,
        mode="reset",
        params={
            "asset_name": "Cup",
            "color_low": (0.30, 0.08, 0.03),
            "color_high": (1.0, 0.65, 0.35),
            "roughness_range": (0.25, 0.85),
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
class SO101ObjectInCupVisionEnvCfg(SO101ObjectInCupEnvCfg):
    scene: ObjectInCupVisionSceneCfg = ObjectInCupVisionSceneCfg(
        # Per-environment USD materials must remain independently authorable.
        num_envs=64,
        env_spacing=0.8,
        replicate_physics=False,
    )
    observations: VisionObservationsCfg = VisionObservationsCfg()
    actions: VisionActionsCfg = VisionActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: VisionEventsCfg = VisionEventsCfg()

    def __post_init__(self):
        require_vision_assets()
        super().__post_init__()
        self.sim.render_interval = self.decimation

    def play_mode(self):
        ManagerBasedRLEnvCfg.play_mode(self)
        self.observations.joint_state.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.scene.ee_frame.debug_vis = True


@configclass
class SO101ObjectInCupVisionFixedEnvCfg(SO101ObjectInCupVisionEnvCfg):
    """Nominal fixed-pose environment used to prove visual learnability first."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_delta.max_delay_steps = 0
        self.observations.joint_state.enable_corruption = False
        self.observations.joint_state.absolute_joint_positions.noise = None
        for group_name in ("wrist", "overhead_1", "overhead_2"):
            term = getattr(self.observations, group_name).rgb
            term.params["randomize"] = False
            term.params["max_delay_steps"] = 0

        self.events.object_material = None
        self.events.cup_material = None
        self.events.object_mass = None
        self.events.actuator_response = None
        self.events.randomize_cameras = None
        self.events.object_visual = None
        self.events.cup_visual = None
        self.events.table_visual = None
        self.events.lighting = None
        self.events.reset_robot.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_layout.params.update(
            object_xy_range_nominal=(0.0, 0.0),
            object_xy_range_full=(0.0, 0.0),
            cup_xy_range_nominal=(0.0, 0.0),
            cup_xy_range_full=(0.0, 0.0),
            object_yaw_range_full=(0.0, 0.0),
        )
