"""Launch the safety-gated local RSL-RL visual-policy node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("model_dir"),
        DeclareLaunchArgument("camera_profile", default_value="dual_overhead"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("max_age_s", default_value="0.2"),
        DeclareLaunchArgument("max_skew_s", default_value="0.05"),
        DeclareLaunchArgument(
            "fwd_topic", default_value="/follower/forward_controller/commands"
        ),
        DeclareLaunchArgument("joints_topic", default_value="/follower/joint_states"),
    ]
    node = Node(
        package="so101_inference",
        executable="rsl_rl_inference_node",
        name="rsl_rl_inference_node",
        parameters=[
            {
                "model_dir": LaunchConfiguration("model_dir"),
                "camera_profile": LaunchConfiguration("camera_profile"),
                "device": LaunchConfiguration("device"),
                "max_age_s": LaunchConfiguration("max_age_s"),
                "max_skew_s": LaunchConfiguration("max_skew_s"),
                "fwd_topic": LaunchConfiguration("fwd_topic"),
                "joints_topic": LaunchConfiguration("joints_topic"),
            }
        ],
        output="screen",
    )
    return LaunchDescription(arguments + [node])
