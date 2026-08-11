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
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from so101_bringup.camera_launch import declare_camera_arguments, include_cameras


def generate_launch_description():

    # --- Launch arguments ---
    hardware_type = LaunchConfiguration("hardware_type")  # real|mock|mujoco
    use_follower = LaunchConfiguration("use_follower")
    leader_ns = LaunchConfiguration("leader_namespace")
    follower_ns = LaunchConfiguration("follower_namespace")
    leader_frame_prefix = LaunchConfiguration("leader_frame_prefix")
    follower_frame_prefix = LaunchConfiguration("follower_frame_prefix")

    leader_usb = LaunchConfiguration("leader_usb_port")
    follower_usb = LaunchConfiguration("follower_usb_port")

    leader_ctrl_cfg = LaunchConfiguration("leader_controller_config_file")
    follower_ctrl_cfg = LaunchConfiguration("follower_controller_config_file")

    leader_rviz = LaunchConfiguration("leader_rviz")
    follower_rviz = LaunchConfiguration("follower_rviz")

    teleop_params_file = LaunchConfiguration("teleop_params_file")
    teleop_delay_s = LaunchConfiguration("teleop_delay_s")

    use_teleop_rviz = LaunchConfiguration("use_teleop_rviz")

    use_rerun = LaunchConfiguration("use_rerun")
    use_rerun_3d = LaunchConfiguration("use_rerun_3d")
    rerun_env_dir = LaunchConfiguration("rerun_env_dir")
    rerun_delay_s = LaunchConfiguration("rerun_delay_s")

    # --- Include leader bringup ---
    leader_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "leader.launch.py"])
        ),
        launch_arguments={
            "namespace": leader_ns,
            "hardware_type": hardware_type,
            "usb_port": leader_usb,
            "frame_prefix": leader_frame_prefix,
            "controller_config_file": leader_ctrl_cfg,
            "use_rviz": leader_rviz,
        }.items(),
    )

    # --- Include follower bringup ---
    follower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "follower.launch.py"])
        ),
        condition=IfCondition(use_follower),
        launch_arguments={
            "namespace": follower_ns,
            "hardware_type": hardware_type,
            "usb_port": follower_usb,
            "frame_prefix": follower_frame_prefix,
            "controller_config_file": follower_ctrl_cfg,
            "use_rviz": follower_rviz,
        }.items(),
    )

    # --- Include teleop launch ---
    teleop_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_teleop"), "launch", "teleop.launch.py"])
        ),
        launch_arguments={
            "leader_namespace": leader_ns,
            "follower_namespace": follower_ns,
            "params_file": teleop_params_file,
        }.items(),
    )

    teleop_start = TimerAction(
        period=teleop_delay_s,
        actions=[teleop_include],
    )

    # --- Include cameras launch ---
    cameras_launch = include_cameras(follower_ns, follower_frame_prefix)

    # --- Include layout launch tf
    layout_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("so101_bringup"), "launch", "layout_tf.launch.py"])
        ),
    )

    # --- Rviz Node

    teleop_rviz = PathJoinSubstitution([FindPackageShare("so101_bringup"), "rviz", "teleop.rviz"])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="teleop_rviz",
        arguments=["-d", teleop_rviz],
        condition=IfCondition(use_teleop_rviz),
        output="screen",
    )

    # --- Launch Rerun

    rerun_bridge_proc = ExecuteProcess(
        cmd=[
            "pixi",
            "run",
            "bridge",
            "--",
            "--camera-profile",
            LaunchConfiguration("camera_profile"),
        ],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        condition=IfCondition(use_rerun),
        output="screen",
    )

    rerun_start = TimerAction(
        period=rerun_delay_s,
        actions=[rerun_bridge_proc],
    )

    # --- Launch Rerun 3D (animated URDF + TF + cameras + plots) ---

    rerun_3d_bridge_proc = ExecuteProcess(
        cmd=[
            "pixi",
            "run",
            "bridge-3d",
            "--",
            "--camera-profile",
            LaunchConfiguration("camera_profile"),
        ],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        condition=IfCondition(use_rerun_3d),
        output="screen",
    )

    rerun_3d_start = TimerAction(
        period=rerun_delay_s,
        actions=[rerun_3d_bridge_proc],
    )

    # --- Defaults for files ---
    default_leader_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "leader_controllers.yaml",
        ]
    )
    default_follower_ctrl_cfg = PathJoinSubstitution(
        [
            FindPackageShare("so101_bringup"),
            "config",
            "ros2_control",
            "follower_controllers.yaml",
        ]
    )
    default_teleop_params = PathJoinSubstitution([FindPackageShare("so101_teleop"), "config", "teleop.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            DeclareLaunchArgument(
                "use_follower",
                default_value="true",
                description="Start the physical/mock follower ros2_control stack. Set false when Isaac Sim is the follower.",
            ),
            DeclareLaunchArgument("leader_namespace", default_value="leader"),
            DeclareLaunchArgument("follower_namespace", default_value="follower"),
            DeclareLaunchArgument("leader_frame_prefix", default_value="leader/"),
            DeclareLaunchArgument("follower_frame_prefix", default_value="follower/"),
            DeclareLaunchArgument("leader_usb_port", default_value="/dev/so101_leader"),
            DeclareLaunchArgument("follower_usb_port", default_value="/dev/so101_follower"),
            DeclareLaunchArgument("leader_controller_config_file", default_value=default_leader_ctrl_cfg),
            DeclareLaunchArgument(
                "follower_controller_config_file",
                default_value=default_follower_ctrl_cfg,
            ),
            DeclareLaunchArgument("leader_rviz", default_value="false"),
            DeclareLaunchArgument("follower_rviz", default_value="false"),
            DeclareLaunchArgument("teleop_params_file", default_value=default_teleop_params),
            DeclareLaunchArgument("teleop_delay_s", default_value="2.0"),
            *declare_camera_arguments(),
            DeclareLaunchArgument("use_teleop_rviz", default_value="true"),
            DeclareLaunchArgument("use_rerun", default_value="false"),
            DeclareLaunchArgument(
                "use_rerun_3d",
                default_value="false",
                description="Launch the 3D Rerun bridge (animated URDF + TF + cameras + plots)",
            ),
            DeclareLaunchArgument(
                "rerun_env_dir",
                # Best: set env var once, no need to pass each run:
                # export SO101_RERUN_ENV_DIR=/abs/path/to/tools/rerun_env
                default_value=EnvironmentVariable("SO101_RERUN_ENV_DIR", default_value=""),
            ),
            DeclareLaunchArgument("rerun_delay_s", default_value=teleop_delay_s),
            leader_launch,
            follower_launch,
            layout_tf_launch,
            cameras_launch,
            rviz_node,
            rerun_start,
            rerun_3d_start,
            teleop_start,
        ]
    )
