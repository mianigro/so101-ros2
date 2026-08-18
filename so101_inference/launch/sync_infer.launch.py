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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("repo_id", description="Hugging Face policy repo ID or local path"),
        DeclareLaunchArgument(
            "setup",
            description=(
                "Required setup: monomanual, monomanual_dual_overhead, or bimanual"
            ),
        ),
        DeclareLaunchArgument("policy_type", default_value="act"),
        DeclareLaunchArgument("task", default_value="Put the green cube in the cup."),
        DeclareLaunchArgument("max_age_s", default_value="0.2"),
        # Per-follower topics; the empty default derives them from the setup
        DeclareLaunchArgument("fwd_topics", default_value="['']"),
        DeclareLaunchArgument("joints_topics", default_value="['']"),
    ]

    node = Node(
        package="so101_inference",
        executable="lerobot_inference_node",
        name="lerobot_inference_node",
        parameters=[
            {
                "repo_id": LaunchConfiguration("repo_id"),
                "setup": LaunchConfiguration("setup"),
                "policy_type": LaunchConfiguration("policy_type"),
                "task": LaunchConfiguration("task"),
                "max_age_s": LaunchConfiguration("max_age_s"),
                "fwd_topics": LaunchConfiguration("fwd_topics"),
                "joints_topics": LaunchConfiguration("joints_topics"),
            }
        ],
        output="screen",
    )

    return LaunchDescription(args + [node])
