"""Setup-driven follower arms + cameras + headless episode recorder + optional rerun.

Generic data-collection / perception / recorder stack.
Task-specific launch files (training, teleop, etc.) should include this.
"""

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
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare
from so101_bringup.camera_launch import declare_setup_arguments


def generate_launch_description():

    # ── Launch arguments ─────────────────────────────────────────
    hardware_type = LaunchConfiguration("hardware_type")
    root_dir = LaunchConfiguration("root_dir")
    experiment_name = LaunchConfiguration("experiment_name")
    task = LaunchConfiguration("task")
    setup = LaunchConfiguration("setup")

    use_rerun = LaunchConfiguration("use_rerun")
    use_rerun_3d = LaunchConfiguration("use_rerun_3d")
    rerun_env_dir = LaunchConfiguration("rerun_env_dir")
    rerun_delay_s = LaunchConfiguration("rerun_delay_s")

    # ── Follower arms + cameras ──────────────────────────────────
    follower_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("so101_bringup"), "launch", "follower_vision.launch.py"
            ])
        ),
        launch_arguments={
            "hardware_type": hardware_type,
            "use_cameras": LaunchConfiguration("use_cameras"),
            "setup": setup,
            "setup_config_file": LaunchConfiguration("setup_config_file"),
            "camera_startup_timeout_s": LaunchConfiguration("camera_startup_timeout_s"),
            "camera_stale_timeout_s": LaunchConfiguration("camera_stale_timeout_s"),
            "use_rviz": "false",
        }.items(),
    )

    # ── Episode recorder (headless) ──────────────────────────────
    recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("episode_recorder"), "launch", "recorder.launch.py"]
            )
        ),
        launch_arguments={
            "setup": setup,
            "setup_config_file": LaunchConfiguration("setup_config_file"),
            "root_dir": root_dir,
            "experiment_name": experiment_name,
            "task": task,
        }.items(),
    )

    # ── Optional rerun bridge ────────────────────────────────────
    rerun_bridge_proc = ExecuteProcess(
        cmd=["pixi", "run", "bridge", "--", "--setup", setup],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        condition=IfCondition(use_rerun),
        output="screen",
    )

    rerun_start = TimerAction(
        period=rerun_delay_s,
        actions=[rerun_bridge_proc],
    )

    # ── Optional 3D rerun bridge (animated URDF + TF + cameras + plots) ──
    rerun_3d_bridge_proc = ExecuteProcess(
        cmd=["pixi", "run", "bridge-3d", "--", "--setup", setup],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        condition=IfCondition(use_rerun_3d),
        output="screen",
    )

    rerun_3d_start = TimerAction(
        period=rerun_delay_s,
        actions=[rerun_3d_bridge_proc],
    )

    # ── Defaults ─────────────────────────────────────────────────
    default_root_dir = PathJoinSubstitution(
        [
            EnvironmentVariable(
                "ROS_HOME",
                default_value=PathJoinSubstitution([EnvironmentVariable("HOME"), ".ros"]),
            ),
            "so101_episodes",
        ]
    )

    return LaunchDescription(
        [
            # Arms + cameras
            DeclareLaunchArgument("hardware_type", default_value="real"),
            *declare_setup_arguments(),
            # Recorder
            DeclareLaunchArgument("root_dir", default_value=default_root_dir),
            DeclareLaunchArgument("experiment_name", default_value="pick_and_place"),
            DeclareLaunchArgument("task", default_value=""),
            # Rerun
            DeclareLaunchArgument("use_rerun", default_value="false"),
            DeclareLaunchArgument(
                "use_rerun_3d",
                default_value="false",
                description="Launch the 3D Rerun bridge (animated URDF + TF + cameras + plots)",
            ),
            DeclareLaunchArgument(
                "rerun_env_dir",
                default_value=EnvironmentVariable("SO101_RERUN_ENV_DIR", default_value=""),
            ),
            DeclareLaunchArgument("rerun_delay_s", default_value="3.0"),
            # Actions
            follower_vision_launch,
            recorder_launch,
            rerun_start,
            rerun_3d_start,
        ]
    )
