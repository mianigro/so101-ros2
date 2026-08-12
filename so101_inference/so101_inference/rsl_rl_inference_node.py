"""Safety-gated ROS 2 inference node for exported visual RSL-RL actors."""

from __future__ import annotations

from functools import partial
import ssl  # noqa: F401  # Preload Pixi OpenSSL before rclpy loads system libcrypto.
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
import torch

from so101_inference.camera_config import (
    camera_topics_for_profile,
    streams_fresh,
    streams_ready,
)
from so101_inference.rsl_rl_policy import (
    EXPECTED_CAMERAS,
    EXPECTED_JOINTS,
    load_policy_manifest,
    preprocess_rgb,
    safe_absolute_targets,
    timestamps_within_skew,
)
from so101_inference.utils import ros2_image_to_numpy


def _stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


class RslRlInferenceNode(Node):
    """Run visual PPO in shadow mode until explicitly armed through SetBool."""

    def __init__(self) -> None:
        super().__init__("rsl_rl_inference_node")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("camera_profile", "dual_overhead")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("max_age_s", 0.2)
        self.declare_parameter("max_skew_s", 0.05)
        self.declare_parameter("fwd_topic", "/follower/forward_controller/commands")
        self.declare_parameter("joints_topic", "/follower/joint_states")

        model_dir_value = str(self.get_parameter("model_dir").value).strip()
        model_dir = Path(model_dir_value)
        camera_profile = str(self.get_parameter("camera_profile").value).strip()
        device_name = str(self.get_parameter("device").value).strip()
        self.max_age_s = float(self.get_parameter("max_age_s").value)
        self.max_skew_s = float(self.get_parameter("max_skew_s").value)
        self.fwd_topic = str(self.get_parameter("fwd_topic").value)
        self.joints_topic = str(self.get_parameter("joints_topic").value)
        if not model_dir_value:
            raise ValueError("model_dir is required")
        if self.max_age_s <= 0.0 or not 0.0 < self.max_skew_s <= 0.05:
            raise ValueError("max_age_s must be positive and max_skew_s must be in (0, 0.05]")
        if not device_name.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("RSL-RL deployment requires the configured CUDA workstation GPU")

        self.manifest = load_policy_manifest(model_dir, camera_profile)
        self.device = torch.device(device_name)
        self.policy = torch.jit.load(
            self.manifest["_policy_path"], map_location=self.device
        ).eval()
        policy_contract = self.manifest["policy"]
        self.delta_scales = np.asarray(
            policy_contract["delta_scales_rad"], dtype=np.float64
        )
        self.joint_limits = np.asarray(
            policy_contract["joint_limits_rad"], dtype=np.float64
        )
        self.safety_margin = float(policy_contract["joint_limit_safety_margin"])

        self.camera_topics = camera_topics_for_profile(camera_profile)
        if tuple(self.camera_topics) != EXPECTED_CAMERAS:
            raise ValueError("dual_overhead ROS camera order changed unexpectedly")
        self.latest_images: dict[str, Image | None] = {
            name: None for name in EXPECTED_CAMERAS
        }
        self.received_at = {name: None for name in EXPECTED_CAMERAS}
        self.source_stamps: dict[str, int | None] = {
            name: None for name in (*EXPECTED_CAMERAS, "joints")
        }
        self.received_at["joints"] = None
        self.latest_joints: np.ndarray | None = None
        self.joint_indices: list[int] | None = None
        self.enabled = False
        self.last_shadow_action: np.ndarray | None = None
        self._last_shadow_log = 0.0

        for name, topic in self.camera_topics.items():
            self.create_subscription(
                Image,
                topic,
                partial(self._camera_callback, name),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState,
            self.joints_topic,
            self._joint_callback,
            qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(Float64MultiArray, self.fwd_topic, 10)
        self.create_service(SetBool, "/so101_rl/set_enabled", self._set_enabled)
        self.create_timer(0.05, self._inference_tick)
        self.get_logger().info(
            "RSL-RL visual policy loaded in SHADOW MODE; use /so101_rl/set_enabled to arm"
        )

    def _camera_callback(self, name: str, message: Image) -> None:
        self.latest_images[name] = message
        self.received_at[name] = self.get_clock().now()
        self.source_stamps[name] = _stamp_ns(message)

    def _joint_callback(self, message: JointState) -> None:
        if self.joint_indices is None:
            by_name = {name: index for index, name in enumerate(message.name)}
            missing = [name for name in EXPECTED_JOINTS if name not in by_name]
            if missing:
                self.get_logger().error(f"JointState missing required joints: {missing}")
                return
            self.joint_indices = [by_name[name] for name in EXPECTED_JOINTS]
        try:
            positions = np.asarray(
                [message.position[index] for index in self.joint_indices],
                dtype=np.float64,
            )
        except IndexError:
            self.get_logger().error("JointState position vector is shorter than its name vector")
            return
        if not np.all(np.isfinite(positions)):
            if self.enabled:
                self._disarm_with_hold("JointState contains NaN or Inf")
            return
        self.latest_joints = positions
        self.received_at["joints"] = self.get_clock().now()
        self.source_stamps["joints"] = _stamp_ns(message)

    def _ready(self) -> bool:
        return (
            streams_ready(EXPECTED_CAMERAS, self.latest_images, self.received_at)
            and self.latest_joints is not None
            and self.received_at["joints"] is not None
        )

    def _fresh_and_synchronized(self) -> bool:
        return streams_fresh(
            self.received_at, self.get_clock().now(), self.max_age_s
        ) and timestamps_within_skew(self.source_stamps, self.max_skew_s)

    def _publish(self, positions: np.ndarray) -> None:
        message = Float64MultiArray()
        message.data = positions.astype(np.float64).tolist()
        self.publisher.publish(message)

    def _disarm_with_hold(self, reason: str) -> None:
        was_enabled = self.enabled
        self.enabled = False
        if was_enabled and self.latest_joints is not None and np.all(
            np.isfinite(self.latest_joints)
        ):
            self._publish(self.latest_joints)
            self.get_logger().error(f"DISARMED; published one hold target: {reason}")
        elif was_enabled:
            self.get_logger().error(f"DISARMED without hold (no valid joint state): {reason}")

    def _set_enabled(self, request: SetBool.Request, response: SetBool.Response):
        if not request.data:
            self._disarm_with_hold("manual disable")
            response.success = True
            response.message = "policy disabled"
            return response
        if not self._ready():
            response.success = False
            response.message = "cannot arm: streams are not ready"
            return response
        if not self._fresh_and_synchronized():
            response.success = False
            response.message = "cannot arm: streams are stale or exceed 50 ms skew"
            return response
        self.enabled = True
        response.success = True
        response.message = "policy armed"
        self.get_logger().warn("RSL-RL policy ARMED and publishing controller targets")
        return response

    def _inference_tick(self) -> None:
        if not self._ready():
            if self.enabled:
                self._disarm_with_hold("input stream unavailable")
            return
        if not self._fresh_and_synchronized():
            if self.enabled:
                self._disarm_with_hold("input stream stale or timestamp skew exceeds 50 ms")
            return
        try:
            images = [
                preprocess_rgb(
                    ros2_image_to_numpy(self.latest_images[name]), self.device
                )
                for name in EXPECTED_CAMERAS
            ]
            joints = torch.from_numpy(self.latest_joints.astype(np.float32)).to(
                self.device
            ).unsqueeze(0)
            with torch.inference_mode():
                action = self.policy(joints, images).squeeze(0).detach().cpu().numpy()
            targets = safe_absolute_targets(
                action,
                self.latest_joints,
                self.delta_scales,
                self.joint_limits,
                self.safety_margin,
            )
        except Exception as error:
            if self.enabled:
                self._disarm_with_hold(f"inference or action validation failed: {error}")
            else:
                self.get_logger().error(f"shadow inference failed: {error}")
            return

        self.last_shadow_action = targets
        if self.enabled:
            self._publish(targets)
        else:
            now = time.monotonic()
            if now - self._last_shadow_log >= 5.0:
                self.get_logger().info("shadow inference healthy; no command published")
                self._last_shadow_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RslRlInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._disarm_with_hold("node shutdown")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
