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

"""Launch intrinsic calibration for one canonical camera role."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_role = LaunchConfiguration("camera_role")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_role",
                description="Required camera role to calibrate",
                choices=["wrist", "overhead_1", "overhead_2"],
            ),
            Node(
                package="so101_camera_calibration",
                executable="camera_intrinsic_calibration_node",
                name="camera_intrinsic_calibration",
                output="screen",
                parameters=[{"camera_role": camera_role}],
            ),
        ]
    )
