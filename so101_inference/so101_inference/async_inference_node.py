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

"""
Thin ROS2 node for SO101 async inference.

Wires ROS2 subscriptions/publishers/timers and delegates all inference
logic to :class:`AsyncInferenceClient`.
"""

from __future__ import annotations

from functools import partial
import ssl  # Preload pixi/conda OpenSSL before rclpy loads system libcrypto.
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray

from so101_inference.async_client import AsyncInferenceClient, ClientCfg
from so101_inference.camera_config import (
    build_lerobot_features,
    camera_subscription_topics,
    camera_topics_for_profile,
    streams_fresh,
    streams_ready,
)
from so101_inference.transport.grpc_transport import GrpcTransport
from so101_inference.utils import ros2_image_to_numpy


class AsyncRos2InferenceClient(Node):
    """Thin ROS2 wrapper that delegates inference to ``AsyncInferenceClient``."""

    def __init__(self) -> None:
        super().__init__("async_ros2_inference_client")

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter("transport_type", "zmq")
        self.declare_parameter("server_address", "127.0.0.1:8090")
        self.declare_parameter("policy_type", "act")
        self.declare_parameter("repo_id", "")
        self.declare_parameter("camera_profile", "")
        self.declare_parameter("policy_device", "cuda")
        self.declare_parameter("actions_per_chunk", 100)
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("fps", 50.0)
        self.declare_parameter("max_age_s", 0.2)
        self.declare_parameter("task", "put the green cube in the cup")
        self.declare_parameter("aggregate_fn_name", "weighted_average")

        self.declare_parameter("fwd_topic", "/follower/forward_controller/commands")
        self.declare_parameter("joints_topic", "/follower/joint_states")

        # When True, subscribe to CompressedImage topics and forward
        # raw JPEG bytes to the server (decoded server-side).
        self.declare_parameter("use_compressed", False)

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

        cfg = ClientCfg(
            server_address=str(self.get_parameter("server_address").value),
            policy_type=str(self.get_parameter("policy_type").value),
            repo_id=str(self.get_parameter("repo_id").value).strip(),
            policy_device=str(self.get_parameter("policy_device").value),
            actions_per_chunk=int(self.get_parameter("actions_per_chunk").value),
            chunk_size_threshold=float(self.get_parameter("chunk_size_threshold").value),
            fps=float(self.get_parameter("fps").value),
            max_age_s=float(self.get_parameter("max_age_s").value),
            task=str(self.get_parameter("task").value),
            aggregate_fn_name=str(self.get_parameter("aggregate_fn_name").value),
        )

        self.fwd_topic = str(self.get_parameter("fwd_topic").value)
        self.joints_topic = str(self.get_parameter("joints_topic").value)
        self.camera_profile = str(self.get_parameter("camera_profile").value).strip()
        if not cfg.repo_id:
            raise ValueError("repo_id is required and must not be empty")
        self.camera_topics = camera_topics_for_profile(self.camera_profile)
        self.arm_joints = list(self.get_parameter("arm_joints").value)

        self._use_compressed = bool(self.get_parameter("use_compressed").value)

        if cfg.fps <= 0:
            self.get_logger().warn(f"Invalid fps={cfg.fps}; forcing 30.0")
            cfg.fps = 30.0

        self.cfg = cfg
        self._log = self.get_logger()

        # --------------------
        # Transport + Client
        # --------------------
        transport_type = str(self.get_parameter("transport_type").value)
        if transport_type == "zmq":
            from so101_inference.transport.zmq_transport import ZmqTransport

            host, port_str = cfg.server_address.rsplit(":", 1)
            zmq_port = int(port_str)
            transport = ZmqTransport(host, port=zmq_port, logger=self.get_logger())
        else:
            transport = GrpcTransport(cfg.server_address, cfg.fps, logger=self.get_logger())

        lerobot_features = build_lerobot_features(self.camera_profile)
        self.client = AsyncInferenceClient(
            transport=transport,
            cfg=cfg,
            lerobot_features=lerobot_features,
            logger=self._log,
        )
        self.client.start()

        # --------------------
        # ROS Runtime state
        # --------------------
        self._latest_camera_data: dict[str, Image | bytes | None] = {
            camera_name: None for camera_name in self.camera_topics
        }
        self._rx_cameras = {camera_name: None for camera_name in self.camera_topics}
        self._rx_joints = None

        self._joint_idx: list[int] | None = None
        self._joint_idx_ready = False
        self._latest_joints_vec: np.ndarray | None = None

        # --------------------
        #  ROS2 Subscribers, Publishers, Timers
        # --------------------
        subscription_topics = camera_subscription_topics(
            self.camera_profile, self._use_compressed
        )
        message_type = CompressedImage if self._use_compressed else Image
        for camera_name, camera_topic in subscription_topics.items():
            self.create_subscription(
                message_type,
                camera_topic,
                partial(self._on_camera_image_cb, camera_name),
                qos_profile_sensor_data,
            )
        if self._use_compressed:
            self._log.info(
                f"📷 Using COMPRESSED images: {list(subscription_topics.values())}"
            )
        self.create_subscription(JointState, self.joints_topic, self._on_joints_cb, qos_profile_sensor_data)

        self.forward_pub = self.create_publisher(Float64MultiArray, self.fwd_topic, 10)

        period = 1.0 / self.cfg.fps
        self.create_timer(period, self.control_loop)

        # Startup logs
        self._log.info("=" * 60)
        self._log.info("  AsyncRos2InferenceClient READY")
        self._log.info("=" * 60)
        self._log.info(f"  server:             {self.cfg.server_address}")
        self._log.info(f"  policy:             {self.cfg.policy_type} | {self.cfg.repo_id}")
        self._log.info(f"  policy_device:      {self.cfg.policy_device}")
        self._log.info(f"  camera_profile:     {self.camera_profile}")
        for camera_name, camera_topic in self.camera_topics.items():
            self._log.info(f"  camera {camera_name}: {camera_topic}")
        self._log.info(f"  joints_topic:       {self.joints_topic}")
        self._log.info(f"  fwd_topic:          {self.fwd_topic}")
        self._log.info(f"  fps:                {self.cfg.fps:.1f}  (period={period * 1000:.1f}ms)")
        self._log.info(f"  actions/chunk:      {self.cfg.actions_per_chunk}")
        self._log.info(f"  chunk_threshold:    {self.cfg.chunk_size_threshold}")
        self._log.info(f"  max_age_s:          {self.cfg.max_age_s}")
        self._log.info(f"  task:               {self.cfg.task}")
        self._log.info("=" * 60)

    # ---------------------------------
    #   ROS Callbacks
    # ---------------------------------

    def _on_camera_image_cb(
        self, camera_name: str, msg: Image | CompressedImage
    ) -> None:
        if self._use_compressed:
            self._latest_camera_data[camera_name] = bytes(msg.data)
        else:
            self._latest_camera_data[camera_name] = msg
        self._rx_cameras[camera_name] = self.get_clock().now()

    def _on_joints_cb(self, msg: JointState):
        if not self._joint_idx_ready:
            if not self._initialize_joint_indices(msg):
                return
        pos = msg.position
        self._latest_joints_vec = np.array([pos[i] for i in self._joint_idx], dtype=np.float32)
        self._rx_joints = self.get_clock().now()

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
            self._log.error(f"JointState missing joints: {missing}. Available: {list(msg.name)}")
            return False
        self._joint_idx = idx
        self._joint_idx_ready = True
        self._log.info(f"Joint mapping initialized: {self.arm_joints} → indices {idx}")
        return True

    # ---------------------------------
    #   Control Loop Helpers
    # ---------------------------------

    def _data_ready(self) -> bool:
        return (
            streams_ready(
                self.camera_topics,
                self._latest_camera_data,
                self._rx_cameras,
            )
            and self._latest_joints_vec is not None
            and self._rx_joints is not None
        )

    def _is_data_fresh(self) -> bool:
        now = self.get_clock().now()

        received_at = {**self._rx_cameras, "joints": self._rx_joints}
        return streams_fresh(received_at, now, self.cfg.max_age_s)

    def _get_data_ages(self) -> dict[str, float]:
        """Return age in ms for each sensor stream."""
        now = self.get_clock().now()
        ages = {}
        for camera_name, received_at in self._rx_cameras.items():
            if received_at is not None:
                ages[camera_name] = (now - received_at).nanoseconds * 1e-6
        if self._rx_joints is not None:
            ages["joints"] = (now - self._rx_joints).nanoseconds * 1e-6
        return ages

    def _build_raw_observation(self) -> dict:
        j = self._latest_joints_vec
        joints_str = " ".join(f"{v:+.4f}" for v in j)

        camera_data = {}
        camera_summaries = []
        for camera_name, data in self._latest_camera_data.items():
            if self._use_compressed:
                camera_data[camera_name] = data
                camera_summaries.append(f"{camera_name}_jpeg={len(data)}B")
            else:
                rgb = ros2_image_to_numpy(data)
                camera_data[camera_name] = rgb
                camera_summaries.append(f"{camera_name}_img={rgb.shape}")
        self._log.debug(
            f"  obs joints: [{joints_str}] | {' '.join(camera_summaries)}"
        )

        return {
            "shoulder_pan.pos": float(j[0]),
            "shoulder_lift.pos": float(j[1]),
            "elbow_flex.pos": float(j[2]),
            "wrist_flex.pos": float(j[3]),
            "wrist_roll.pos": float(j[4]),
            "gripper.pos": float(j[5]),
            **camera_data,
            "task": self.cfg.task,
        }

    # ---------------------------------
    #   Control Loop
    # ---------------------------------

    def control_loop(self):
        loop_start = time.perf_counter()
        self.client.increment_control_loop()

        # Require data
        if not self._data_ready():
            if self.client._control_loop_count % 100 == 0:
                camera_status = " ".join(
                    f"{name}={'✓' if data is not None else '✗'}"
                    for name, data in self._latest_camera_data.items()
                )
                self._log.warn(
                    f"⏳ Waiting for sensor data... "
                    f"{camera_status} "
                    f"joints={'✓' if self._latest_joints_vec is not None else '✗'}"
                )
            return

        # Require data fresh
        if not self._is_data_fresh():
            ages = self._get_data_ages()
            ages_str = " ".join(f"{k}={v:.0f}ms" for k, v in ages.items())
            self._log.warn(f"⚠️ Stale sensor data (max_age={self.cfg.max_age_s * 1000:.0f}ms) | {ages_str}")
            return

        # (1) execute next action if any
        if self.client.actions_available():
            action_np = self.client.pop_action()
            if action_np is not None:
                msg = Float64MultiArray()
                msg.data = action_np.tolist()
                self.forward_pub.publish(msg)
        else:
            if self.client._control_loop_count % 30 == 0:
                self._log.debug("⏸️ No actions in queue to execute")

        # (2) send observations when queue is "low enough"
        if self.client.ready_to_send():
            ages = self._get_data_ages()
            ages_str = " ".join(f"{k}={v:.0f}ms" for k, v in ages.items())
            self._log.info(f"  sensor ages: {ages_str}")

            raw_obs = self._build_raw_observation()
            self.client.submit_observation(raw_obs)

        # (3) periodic summary
        self.client.maybe_log_summary()

        loop_ms = (time.perf_counter() - loop_start) * 1000
        if loop_ms > (1000.0 / self.cfg.fps) * 1.5:
            self._log.warn(f"⚠️ Control loop SLOW: {loop_ms:.1f}ms (budget={1000.0 / self.cfg.fps:.1f}ms)")

    # ---------------------------------
    #   Shutdown
    # ---------------------------------

    def destroy_node(self):
        self._log.info("🛑 Shutting down AsyncRos2InferenceClient...")
        self.client.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AsyncRos2InferenceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
