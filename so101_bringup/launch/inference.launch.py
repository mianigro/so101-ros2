"""Inference: setup-driven follower arms + cameras."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from so101_bringup.arms_launch import follower_arms_action
from so101_bringup.camera_launch import declare_setup_arguments, include_cameras
from so101_bringup.rerun_launch import declare_rerun_arguments, rerun_bridge_actions


def generate_launch_description():
    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    setup = LaunchConfiguration("setup")

    # Inference toggles + policy params
    use_inference = LaunchConfiguration("use_inference")
    inference_delay_s = LaunchConfiguration("inference_delay_s")

    repo_id = LaunchConfiguration("repo_id")
    policy_type = LaunchConfiguration("policy_type")
    task = LaunchConfiguration("task")
    rerun_env_dir = LaunchConfiguration("rerun_env_dir")

    # --- Setup-driven follower arms + cameras ---
    arms = follower_arms_action()
    cameras_launch = include_cameras()

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
            ["setup:=", setup],
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
    rerun_start = rerun_bridge_actions(setup, also_requires_dir=use_inference)

    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            *declare_setup_arguments(),
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
            *declare_rerun_arguments(),
            arms,
            cameras_launch,
            rerun_start,
            inference_start,
        ]
    )
