#!/usr/bin/env python3
"""Run an Isaac Sim SO-101 follower controlled by the existing ROS 2 teleop topic."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XACRO_PATH = REPO_ROOT / "so101_description" / "urdf" / "so101_arm.urdf.xacro"
DESCRIPTION_PATH = REPO_ROOT / "so101_description"
DEFAULT_ASSET_DIR = REPO_ROOT / "build" / "isaacsim_so101"

ROBOT_PRIM_PATH = "/World/SO101"
ARTICULATION_PRIM_PATH = f"{ROBOT_PRIM_PATH}/Geometry"
ACTION_GRAPH_PATH = "/SO101_ROS2_ActionGraph"
COMMAND_TOPIC = "/follower/forward_controller/commands"
JOINT_STATE_TOPIC = "/follower/joint_states"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the SO-101 follower and connect it to the Isaac Sim ROS 2 bridge."
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Generated URDF/USD cache directory (default: repo build directory).",
    )
    parser.add_argument(
        "--rebuild-asset",
        action="store_true",
        help="Rebuild the generated SO-101 USD asset from the current Xacro and meshes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Expand and validate the follower URDF without starting Isaac Sim.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without its GUI.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Exit after this many frames; zero keeps running until Isaac Sim closes.",
    )
    parser.add_argument(
        "--publish-clock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish Isaac Sim time on /clock.",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=120.0,
        help="Physics update frequency.",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=100.0,
        help="Initial position-drive stiffness in Nm/rad.",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=2.0,
        help="Initial position-drive damping in Nm*s/rad.",
    )
    args, _ = parser.parse_known_args()
    if args.physics_hz <= 0:
        parser.error("--physics-hz must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.joint_stiffness < 0 or args.joint_damping < 0:
        parser.error("joint stiffness and damping must be non-negative")
    return args


def expand_follower_urdf(asset_dir: Path) -> Path:
    xacro = shutil.which("xacro")
    if xacro is None:
        raise RuntimeError("xacro was not found; source /opt/ros/jazzy/setup.bash and this workspace first")

    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = asset_dir / "so101_follower.urdf"
    temporary_path = asset_dir / "so101_follower.urdf.tmp"
    command = [
        xacro,
        str(XACRO_PATH),
        "variant:=follower",
        "use_ros2_control:=false",
        "add_transmissions:=false",
    ]
    with temporary_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"xacro failed:\n{result.stderr.strip()}")
    temporary_path.replace(output_path)
    validate_follower_urdf(output_path)
    return output_path


def validate_follower_urdf(urdf_path: Path) -> None:
    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot" or robot.attrib.get("name") != "so101_arm":
        raise RuntimeError(f"unexpected robot in generated URDF: {robot.tag} {robot.attrib.get('name')}")
    if robot.find("ros2_control") is not None:
        raise RuntimeError("generated simulation URDF unexpectedly contains ros2_control")

    movable_joints = {
        joint.attrib["name"]
        for joint in robot.findall("joint")
        if joint.attrib.get("type") in {"continuous", "prismatic", "revolute"}
    }
    if movable_joints != set(JOINT_NAMES):
        raise RuntimeError(
            "generated URDF joint mismatch: "
            f"missing={sorted(set(JOINT_NAMES) - movable_joints)}, "
            f"unexpected={sorted(movable_joints - set(JOINT_NAMES))}"
        )
    if not robot.findall(".//collision") or not robot.findall(".//inertial"):
        raise RuntimeError("generated URDF must contain collision and inertial data")


def newest_description_mtime() -> float:
    inputs = list((DESCRIPTION_PATH / "urdf").rglob("*.xacro"))
    inputs.extend((DESCRIPTION_PATH / "meshes").glob("*.stl"))
    return max(path.stat().st_mtime for path in inputs)


def run_isaac_sim(args: argparse.Namespace, urdf_path: Path) -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    exit_code = 0
    try:
        import omni.graph.core as og
        import omni.kit.app
        import usdrt.Sdf
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.experimental.objects import DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import Articulation
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        for extension in (
            "omni.scene.optimizer.core",
            "isaacsim.robot.schema",
            "isaacsim.asset.importer.urdf",
            "isaacsim.ros2.bridge",
        ):
            extension_manager.set_extension_enabled_immediate(extension, True)
        simulation_app.update()

        usd_path = import_or_reuse_asset(args, urdf_path, URDFImporter, URDFImporterConfig)

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        light = DistantLight("/World/DistantLight")
        light.set_intensities(300)
        stage_utils.add_reference_to_stage(usd_path=str(usd_path), path=ROBOT_PRIM_PATH)
        set_camera_view(
            eye=[0.65, 0.65, 0.45],
            target=[0.0, 0.0, 0.16],
            camera_prim_path="/OmniverseKit_Persp",
        )
        simulation_app.update()

        articulation = Articulation(ARTICULATION_PRIM_PATH)
        imported_joints = list(articulation.dof_names)
        if set(imported_joints) != set(JOINT_NAMES):
            raise RuntimeError(
                f"imported articulation joint mismatch: expected={JOINT_NAMES}, actual={imported_joints}"
            )

        create_ros_action_graph(og, simulation_app, usdrt.Sdf, args.publish_clock)
        SimulationManager.setup_simulation(dt=1.0 / args.physics_hz, device="cpu")

        domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
        rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp (Isaac Sim default)")
        print(f"SO-101 USD: {usd_path}", flush=True)
        print(f"Imported DOF order: {imported_joints}", flush=True)
        print(f"ROS_DOMAIN_ID={domain_id}; RMW_IMPLEMENTATION={rmw}", flush=True)
        print(f"Subscribing: {COMMAND_TOPIC} [std_msgs/msg/Float64MultiArray]", flush=True)
        print(f"Publishing: {JOINT_STATE_TOPIC} [sensor_msgs/msg/JointState]", flush=True)

        app_utils.play()
        simulation_app.update()
        frame_count = 0
        while simulation_app.is_running():
            simulation_app.update()
            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    except Exception as error:
        exit_code = 1
        print(f"Isaac Sim setup/runtime error: {error!r}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close(exit_code=exit_code)


def import_or_reuse_asset(args, urdf_path, importer_type, config_type) -> Path:
    asset_dir = args.asset_dir.resolve()
    robot_name = urdf_path.stem
    robot_output_dir = asset_dir / robot_name
    expected_usd = robot_output_dir / f"{robot_name}.usda"

    if robot_output_dir.is_symlink():
        raise RuntimeError(f"refusing to use a symlinked Isaac Sim asset directory: {robot_output_dir}")
    if args.rebuild_asset and robot_output_dir.exists():
        shutil.rmtree(robot_output_dir)
    if expected_usd.exists():
        if expected_usd.stat().st_mtime < newest_description_mtime():
            raise RuntimeError("cached SO-101 USD is stale; rerun with --rebuild-asset")
        return expected_usd
    if robot_output_dir.exists():
        raise RuntimeError(f"incomplete generated asset directory: {robot_output_dir}; rerun with --rebuild-asset")

    config = config_type(
        urdf_path=str(urdf_path),
        usd_path=str(asset_dir),
        merge_fixed_joints=True,
        merge_mesh=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        ros_package_paths=[{"name": "so101_description", "path": str(DESCRIPTION_PATH)}],
        robot_type="Manipulator",
        fix_base=True,
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=args.joint_stiffness,
        override_joint_damping=args.joint_damping,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    usd_path = Path(importer_type(config=config).import_urdf()).resolve()
    if not usd_path.is_file():
        raise RuntimeError(f"URDF importer did not produce the expected USD file: {usd_path}")
    return usd_path


def create_ros_action_graph(og, simulation_app, sdf, publish_clock: bool) -> None:
    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("CommandQoS", "isaacsim.ros2.bridge.ROS2QoSProfile"),
        ("CommandSubscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "CommandSubscriber.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
        ("ROS2Context.outputs:context", "CommandSubscriber.inputs:context"),
        ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
        ("CommandQoS.outputs:qosProfile", "CommandSubscriber.inputs:qosProfile"),
        ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
        ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
        ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
        ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
        ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
        ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
        ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
        ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
    ]
    values = [
        ("CommandQoS.inputs:history", "keepLast"),
        ("CommandQoS.inputs:depth", 10),
        ("CommandQoS.inputs:reliability", "reliable"),
        ("CommandQoS.inputs:durability", "volatile"),
        ("CommandSubscriber.inputs:topicName", COMMAND_TOPIC),
        ("ArticulationController.inputs:robotPath", ARTICULATION_PRIM_PATH),
        ("ArticulationController.inputs:jointNames", JOINT_NAMES),
        ("ReadJointState.inputs:prim", [sdf.Path(ARTICULATION_PRIM_PATH)]),
        ("PublishJointState.inputs:topicName", JOINT_STATE_TOPIC),
        ("PublishJointState.inputs:queueSize", 10),
    ]

    if publish_clock:
        nodes.extend(
            [
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ]
        )
        connections.extend(
            [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ]
        )
        values.append(("PublishClock.inputs:topicName", "/clock"))

    og.Controller.edit(
        {"graph_path": ACTION_GRAPH_PATH, "evaluator_name": "execution"},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connections, keys.SET_VALUES: values},
    )

    subscriber = og.Controller.node(f"{ACTION_GRAPH_PATH}/CommandSubscriber")
    for attribute, value in (
        ("inputs:messagePackage", "std_msgs"),
        ("inputs:messageSubfolder", "msg"),
        ("inputs:messageName", "Float64MultiArray"),
    ):
        og.Controller.attribute(attribute, subscriber).set(value)
        simulation_app.update()

    try:
        controller = og.Controller.node(f"{ACTION_GRAPH_PATH}/ArticulationController")
        og.Controller.connect(
            og.Controller.attribute("outputs:data", subscriber),
            og.Controller.attribute("inputs:positionCommand", controller),
        )
    except Exception as error:
        raise RuntimeError(
            "Isaac Sim did not create Float64MultiArray outputs:data; direct bridge smoke test failed"
        ) from error


def main() -> int:
    args = parse_args()
    args.asset_dir = args.asset_dir.resolve()
    urdf_path = expand_follower_urdf(args.asset_dir)
    print(f"Validated follower URDF: {urdf_path}")
    if args.validate_only:
        return 0
    if os.environ.get("ROS_DISTRO") not in {"jazzy", "humble"}:
        raise RuntimeError("source ROS 2 Jazzy or Humble before starting Isaac Sim")
    run_isaac_sim(args, urdf_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
