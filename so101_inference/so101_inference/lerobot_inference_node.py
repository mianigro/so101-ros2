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

"""SO101 LeRobot ROS2 inference node."""

from copy import copy
from functools import partial
import ssl  # Preload pixi/conda OpenSSL before rclpy loads system libcrypto.
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

import torch

from so101_inference import CONTROL_FREQUENCY_HZ
from so101_inference.setup_config import (
    camera_topics_for_setup,
    streams_fresh,
    streams_ready,
    validate_policy_input_features,
)
from so101_inference.utils import ros2_image_to_numpy

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors


class LeRobotInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_inference_node")

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter("repo_id", "")
        self.declare_parameter("setup", "")
        self.declare_parameter("policy_type", "act")
        self.declare_parameter("task", "Put the green cube in the cup.")
        self.declare_parameter("max_age_s", 0.2)

        # Per-follower topics; empty entries are derived from the setup.
        self.declare_parameter("joints_topics", [""])
        self.declare_parameter("fwd_topics", [""])

        self.declare_parameter(
            "arm_joints",
            [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
        )

        # Read parameters
        self.repo_id = str(self.get_parameter("repo_id").value).strip()
        self.setup = str(self.get_parameter("setup").value).strip()
        self.policy_type = str(self.get_parameter("policy_type").value)
        self.task = str(self.get_parameter("task").value)
        self.fps = CONTROL_FREQUENCY_HZ
        self.max_age_s = float(self.get_parameter("max_age_s").value)

        if not self.repo_id:
            raise ValueError("repo_id is required and must not be empty")
        self.camera_topics = camera_topics_for_setup(self.setup)
        from rosbag_to_lerobot.setups import command_topics, joint_state_topics

        self.joints_topics = self._resolve_topic_list(
            self.get_parameter("joints_topics").value, joint_state_topics(self.setup)
        )
        self.fwd_topics = self._resolve_topic_list(
            self.get_parameter("fwd_topics").value, command_topics(self.setup)
        )
        if len(self.joints_topics) != len(self.fwd_topics):
            raise ValueError("joints_topics and fwd_topics must have equal length")

        self.arm_joints = list(self.get_parameter("arm_joints").value)

        # --------------------
        # Policy Setup
        # --------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Using device: {self.device}")
        self.get_logger().info(f"Loading LeRobot policy from repo_id: {self.repo_id}")
        config = PreTrainedConfig.from_pretrained(self.repo_id)
        validate_policy_input_features(config.input_features, self.setup)
        # config.n_action_steps = 50
        # config.temporal_ensemble_coeff = 0.01
        policy_class = get_policy_class(self.policy_type)
        self.policy = policy_class.from_pretrained(self.repo_id, config=config).to(self.device)
        self.policy.eval()
        self.policy.reset()

        # Load preprocessor/postprocessor (contains normalization stats from training)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.repo_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        # --------------------
        # Runtime state
        # --------------------

        self._latest_camera_images: dict[str, Image | None] = {
            camera_name: None for camera_name in self.camera_topics
        }
        self._rx_cameras = {camera_name: None for camera_name in self.camera_topics}

        # Per-follower joint caches; the concatenated vector follows the setup order
        self._latest_joints_vecs: list[np.ndarray | None] = [None] * len(self.joints_topics)
        self._rx_joints: list = [None] * len(self.joints_topics)

        # Joint ordering cache (shared: every follower broadcasts the same names)
        self._joint_idx: list[int] | None = None
        self._joint_idx_ready = False

        # --------------------
        #  ROS2 Subscribers, Publishers, Timers
        # --------------------
        for camera_name, camera_topic in self.camera_topics.items():
            self.create_subscription(
                Image,
                camera_topic,
                partial(self._on_camera_image_cb, camera_name),
                qos_profile_sensor_data,
            )
        for index, joints_topic in enumerate(self.joints_topics):
            self.create_subscription(
                JointState,
                joints_topic,
                partial(self._on_joints_cb, index),
                qos_profile_sensor_data,
            )

        self.forward_pubs = [
            self.create_publisher(Float64MultiArray, fwd_topic, 10)
            for fwd_topic in self.fwd_topics
        ]

        # --------------------
        # Timing
        # --------------------
        self._inference_count = 0
        self._chunk_count = 0

        # TIMER LOOP
        period = 1.0 / self.fps
        self.create_timer(period, self.inference_loop)

        # Startup logs
        self.get_logger().info("LeRobotInferenceNode READY")
        self.get_logger().info(f"  setup:             {self.setup}")
        for camera_name, camera_topic in self.camera_topics.items():
            self.get_logger().info(f"  camera {camera_name}: {camera_topic}")
        for joints_topic, fwd_topic in zip(self.joints_topics, self.fwd_topics):
            self.get_logger().info(f"  arm topics:         {joints_topic} -> {fwd_topic}")
        self.get_logger().info(f"  fps:                {self.fps:.1f}")
        self.get_logger().info(f"  max_age_s:          {self.max_age_s:.3f}")

    @staticmethod
    def _resolve_topic_list(configured, setup_topics) -> list[str]:
        """Use setup-derived topics unless every configured entry is set."""
        values = [str(value).strip() for value in configured]
        if values and all(values):
            return values
        return list(setup_topics)

    def _on_camera_image_cb(self, camera_name: str, msg: Image):
        self._latest_camera_images[camera_name] = msg
        self._rx_cameras[camera_name] = self.get_clock().now()

    def _on_joints_cb(self, arm_index: int, msg: JointState):
        # Initialize mapping once (or retry until it works)
        if not self._joint_idx_ready:
            if not self._initialize_joint_indices(msg):
                return

        # Cache ordered joints
        pos = msg.position
        self._latest_joints_vecs[arm_index] = np.array(
            [pos[i] for i in self._joint_idx], dtype=np.float32
        )
        self._rx_joints[arm_index] = self.get_clock().now()

    def _initialize_joint_indices(self, msg: JointState) -> bool:
        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        idx = []
        missing = []
        for j in self.arm_joints:
            if j not in name_to_idx:
                missing.append(j)
            else:
                idx.append(name_to_idx[j])

        if missing:
            self.get_logger().error(f"JointState missing joints: {missing}. Available: {list(msg.name)}")
            return False

        self._joint_idx = idx
        self._joint_idx_ready = True
        self.get_logger().info(f"Initialized joint indices for {len(idx)} joints: {self.arm_joints}")
        return True

    def _data_ready(self) -> bool:
        return (
            streams_ready(
                self.camera_topics,
                self._latest_camera_images,
                self._rx_cameras,
            )
            and all(vec is not None for vec in self._latest_joints_vecs)
            and all(rx is not None for rx in self._rx_joints)
        )

    def _is_data_fresh(self) -> bool:
        now = self.get_clock().now()

        received_at = {**self._rx_cameras}
        for index, rx in enumerate(self._rx_joints):
            received_at[f"joints_{index}"] = rx
        return streams_fresh(received_at, now, self.max_age_s)

    def _build_observation(self) -> dict:
        """Return raw RGB camera images and the concatenated joint state."""
        observation = {
            "observation.state": np.concatenate(self._latest_joints_vecs),
            "task": self.task,
        }
        for camera_name, image in self._latest_camera_images.items():
            observation[f"observation.images.{camera_name}"] = ros2_image_to_numpy(image)
        return observation

    def inference_loop(self):
        # Require data
        if not self._data_ready():
            return
        # Require data fresh
        if not self._is_data_fresh():
            return

        t0 = time.perf_counter()

        observation = self._build_observation()

        with torch.inference_mode():
            # Convert numpy → tensor, normalize images, add batch dim, move to device
            obs = copy(observation)
            for name in obs:
                if isinstance(obs[name], np.ndarray):
                    obs[name] = torch.from_numpy(obs[name])
                if "image" in name:
                    obs[name] = obs[name].float() / 255.0
                    obs[name] = obs[name].permute(2, 0, 1).contiguous() # HWC => CHW
                # obs[name] = obs[name].unsqueeze(0).to(self.device) # done in preprocessor

            obs = self.preprocessor(obs)
            action = self.policy.select_action(obs)
            action = self.postprocessor(action)

        # Remove batch dimension and convert to numpy
        action = action.squeeze(0).cpu().numpy()

        # Split the concatenated action per follower and publish each arm's slice
        joints_per_arm = len(action) // len(self.forward_pubs)
        for index, publisher in enumerate(self.forward_pubs):
            msg = Float64MultiArray()
            msg.data = action[index * joints_per_arm:(index + 1) * joints_per_arm]
            publisher.publish(msg)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._inference_count += 1

        # n_action_steps actions per chunk; log once per chunk
        n = self.policy.config.n_action_steps
        if self._inference_count % n == 1 or n == 1:
            self._chunk_count += 1
            self._chunk_forward_ms = elapsed_ms
            self._chunk_start = time.perf_counter()
        elif self._inference_count % n == 0:
            chunk_wall_ms = (time.perf_counter() - self._chunk_start) * 1000
            self.get_logger().info(
                f"⏱️ chunk #{self._chunk_count} | forward={self._chunk_forward_ms:.1f}ms | "
                f"{n} actions in {chunk_wall_ms:.1f}ms wall"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeRobotInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
