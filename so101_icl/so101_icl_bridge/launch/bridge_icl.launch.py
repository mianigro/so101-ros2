# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the ICL bridge node with its mission parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("api_base", default_value=""),
        DeclareLaunchArgument("workspace", default_value=""),
        DeclareLaunchArgument("mission_id", default_value=""),
        DeclareLaunchArgument("demo_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("demo_port", default_value="8661"),
        DeclareLaunchArgument("k", default_value="4"),
        DeclareLaunchArgument("k_max", default_value="4"),
        DeclareLaunchArgument("frames_per_demo", default_value="6"),
        DeclareLaunchArgument("stats_path", default_value=""),
        DeclareLaunchArgument("poll_period_s", default_value="5.0"),
        DeclareLaunchArgument("terminal_msg_type", default_value="std_msgs/msg/Empty"),
    ]
    node = Node(
        package="so101_icl_bridge",
        executable="bridge_icl_node",
        name="bridge_icl_node",
        output="screen",
        parameters=[
            {
                "api_base": LaunchConfiguration("api_base"),
                "workspace": LaunchConfiguration("workspace"),
                "mission_id": LaunchConfiguration("mission_id"),
                "demo_host": LaunchConfiguration("demo_host"),
                "demo_port": LaunchConfiguration("demo_port"),
                "k": LaunchConfiguration("k"),
                "k_max": LaunchConfiguration("k_max"),
                "frames_per_demo": LaunchConfiguration("frames_per_demo"),
                "stats_path": LaunchConfiguration("stats_path"),
                "poll_period_s": LaunchConfiguration("poll_period_s"),
                "terminal_msg_type": LaunchConfiguration("terminal_msg_type"),
            }
        ],
    )
    return LaunchDescription(args + [node])
