"""Inference: follower arm + cameras."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from so101_bringup.camera_launch import declare_camera_arguments, include_cameras
from so101_bringup.rerun_launch import declare_rerun_arguments, rerun_bridge_actions


def generate_launch_description():
    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    follower_ns = LaunchConfiguration("follower_namespace")
    follower_frame_prefix = LaunchConfiguration("follower_frame_prefix")
    follower_usb = LaunchConfiguration("follower_usb_port")

    follower_ctrl_cfg = LaunchConfiguration("follower_controller_config_file")

    # Inference toggles + policy params
    use_inference = LaunchConfiguration("use_inference")
    inference_delay_s = LaunchConfiguration("inference_delay_s")

    repo_id = LaunchConfiguration("repo_id")
    policy_type = LaunchConfiguration("policy_type")
    task = LaunchConfiguration("task")
    camera_profile = LaunchConfiguration("camera_profile")
    # device = LaunchConfiguration("device")

    rerun_env_dir = LaunchConfiguration("rerun_env_dir")

    # --- Include follower bringup ---
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("so101_bringup"), "launch", "follower.launch.py"]
            )
        ),
        launch_arguments={
            "namespace": follower_ns,
            "hardware_type": hardware_type,
            "usb_port": follower_usb,
            "frame_prefix": follower_frame_prefix,
            "controller_config_file": follower_ctrl_cfg,
            "use_rviz": "false",
        }.items(),
    )

    # --- Include cameras launch ---
    cameras_launch = include_cameras(follower_ns, follower_frame_prefix)

    # --- Inference node ---
    # Inference process: runs inside pixi env
    inference_proc = ExecuteProcess(
        cmd=[
            "pixi",
            "run",
            "-e",
            "lerobot",
            "infer",
            "--",
            "--ros-args",
            "-p",
            ["repo_id:=", repo_id],
            "-p",
            ["policy_type:=", policy_type],
            "-p",
            ["task:=", task],
            "-p",
            ["camera_profile:=", camera_profile],
            # "-p",
            # ["device:=", device],
        ],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        output="screen",
        condition=IfCondition(use_inference),
    )

    inference_start = TimerAction(
        period=inference_delay_s,
        actions=[inference_proc],
        condition=IfCondition(use_inference),
    )

    # --- Launch Rerun (validated 2D/3D bridges; also covers the pixi cwd) ---
    rerun_start = rerun_bridge_actions(camera_profile, also_requires_dir=use_inference)

    # --- Defaults for files ---
    default_follower_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "follower_controllers.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            DeclareLaunchArgument("follower_namespace", default_value="follower"),
            DeclareLaunchArgument("follower_frame_prefix", default_value="follower/"),
            DeclareLaunchArgument(
                "follower_usb_port", default_value="/dev/so101_follower"
            ),
            DeclareLaunchArgument(
                "follower_controller_config_file",
                default_value=default_follower_ctrl_cfg,
            ),
            *declare_camera_arguments(),
            DeclareLaunchArgument("use_inference", default_value="false"),
            DeclareLaunchArgument("inference_delay_s", default_value="2.0"),
            DeclareLaunchArgument(
                "repo_id", description="Required Hugging Face policy repo ID or local path"
            ),
            DeclareLaunchArgument(
                "policy_type",
                default_value="act",
                description=(
                    "Policy architecture registered with LeRobot 0.6.1 "
                    "(act, smolvla, diffusion, pi0, pi05, xvla, vqbet, tdmpc, groot, ...). "
                    "ACT/SmolVLA are recommended for this sync node; heavy VLAs (pi05, xvla) "
                    "are better served by the async node + policy_server."
                ),
            ),
            DeclareLaunchArgument(
                "task",
                default_value="Put the green cube in the cup.",
                description="Runtime task instruction; override with your dataset's task text",
            ),
            # DeclareLaunchArgument("device", default_value="cuda"),
            *declare_rerun_arguments(),
            follower_launch,
            cameras_launch,
            rerun_start,
            inference_start,
        ]
    )

