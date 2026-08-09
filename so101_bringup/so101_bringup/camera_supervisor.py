"""Fail-fast watchdog for the configured ROS camera image streams."""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from so101_bringup.camera_config import evaluate_camera_streams, intrinsics_are_valid


def camera_info_is_valid(msg: CameraInfo) -> bool:
    """Return whether a CameraInfo message contains usable pinhole intrinsics."""
    return intrinsics_are_valid(msg.width, msg.height, msg.k)


class CameraSupervisor(Node):
    def __init__(self) -> None:
        super().__init__("camera_supervisor")
        self.declare_parameter("camera_names", [])
        self.declare_parameter("camera_topics", [])
        self.declare_parameter("camera_info_topics", [])
        self.declare_parameter("require_camera_info", True)
        self.declare_parameter("startup_timeout_s", 10.0)
        self.declare_parameter("stale_timeout_s", 1.0)
        names = list(self.get_parameter("camera_names").value)
        topics = list(self.get_parameter("camera_topics").value)
        info_topics = list(self.get_parameter("camera_info_topics").value)
        self._require_camera_info = bool(self.get_parameter("require_camera_info").value)
        self._startup_timeout = float(self.get_parameter("startup_timeout_s").value)
        self._stale_timeout = float(self.get_parameter("stale_timeout_s").value)
        if not names or len(names) != len(topics) or len(names) != len(info_topics):
            raise ValueError(
                "camera_names, camera_topics, and camera_info_topics must be non-empty and have equal length"
            )
        if self._startup_timeout <= 0 or self._stale_timeout <= 0:
            raise ValueError("camera supervisor timeouts must be positive")
        self._started_at = time.monotonic()
        self._last_image: dict[str, float | None] = {name: None for name in names}
        self._last_info: dict[str, float | None] = {name: None for name in names}
        self._failed = False
        self._subscriptions = []
        for name, topic, info_topic in zip(names, topics, info_topics, strict=True):
            self._subscriptions.append(self.create_subscription(
                Image, topic, lambda _msg, camera=name: self._on_image(camera), qos_profile_sensor_data
            ))
            if self._require_camera_info:
                self._subscriptions.append(self.create_subscription(
                    CameraInfo,
                    info_topic,
                    lambda msg, camera=name: self._on_camera_info(camera, msg),
                    qos_profile_sensor_data,
                ))
        self.create_timer(0.1, self._check)
        self.get_logger().info(f"supervising camera streams: {dict(zip(names, topics, strict=True))}")

    def _on_image(self, camera: str) -> None:
        self._last_image[camera] = time.monotonic()

    def _on_camera_info(self, camera: str, msg: CameraInfo) -> None:
        if not camera_info_is_valid(msg):
            self._fail(f"camera {camera} published invalid/zero intrinsics")
            return
        self._last_info[camera] = time.monotonic()

    def _fail(self, reason: str) -> None:
        if self._failed:
            return
        self._failed = True
        self.get_logger().fatal(reason)
        rclpy.shutdown()

    def _check(self) -> None:
        now = time.monotonic()
        missing, stale = evaluate_camera_streams(
            self._last_image,
            self._last_info,
            now,
            self._stale_timeout,
            self._require_camera_info,
        )
        if missing:
            if now - self._started_at > self._startup_timeout:
                self._fail(f"camera startup timeout; no images from: {', '.join(missing)}")
            return
        if stale:
            self._fail(f"camera streams stale for more than {self._stale_timeout:.3f}s: {', '.join(stale)}")


def main() -> int:
    rclpy.init()
    node: CameraSupervisor | None = None
    try:
        node = CameraSupervisor()
        rclpy.spin(node)
        return 1 if node._failed else 0
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(f"camera supervisor failed: {exc}")
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
