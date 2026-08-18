#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray

from so101_setups import (
    SETUP_CAMERA_NAMES,
    command_topics,
    image_topics,
    joint_state_topics,
    follower_label,
)

# LeRobot-style constants
OBS_STR = "observation"
ACTION_STR = "action"


def stamp_to_datetime64(stamp) -> np.datetime64:
    t = Time.from_msg(stamp)
    return np.datetime64(t.nanoseconds, "ns")


def media_type_from_compressed_format(fmt: str) -> Optional[str]:
    f = (fmt or "").lower()
    # Common ROS compressed formats look like "jpeg", "png" or "jpeg compressed bgr8"
    if "jpeg" in f or "jpg" in f:
        return "image/jpeg"
    if "png" in f:
        return "image/png"
    return None


def rgb8_to_numpy(img: Image) -> np.ndarray:
    arr = np.frombuffer(img.data, dtype=np.uint8)
    return arr.reshape(img.height, img.width, 3)  # RGB


def log_scalar(path: str, value: float) -> None:
    rr.log(path, rr.Scalars(value))


@dataclass
class Topics:
    cameras: dict[str, str]
    # (topic, dataset label prefix) per follower arm, in setup order
    joint_states: list[tuple[str, str]]
    forward_commands: list[tuple[str, str]]


class So101Ros2ToRerun(Node):
    def __init__(
        self,
        topics: Topics,
        cmd_joint_order: list[str],
        clear_state_gap_s: float = 2.0,
    ) -> None:
        super().__init__("so101_ros2_to_rerun")
        self._cmd_joint_order = list(cmd_joint_order)
        # Timestamp reference for unstamped messages (Float64MultiArray).
        # Stamped callbacks write here; the action callback reads it.
        # Protected by _time_lock for thread-safety.
        self._time_lock = threading.Lock()
        self._last_ros_time: np.datetime64 | None = None
        self._last_action_time = np.datetime64(0, "ns")
        self._clear_state_gap = np.timedelta64(max(0, int(clear_state_gap_s * 1e9)), "ns")

        # Separate callback groups so heavy-ish callbacks don't block each other.
        self._cg_images = ReentrantCallbackGroup()
        self._cg_joints = ReentrantCallbackGroup()
        self._cg_cmd = ReentrantCallbackGroup()

        for camera_name, camera_topic in topics.cameras.items():
            self._subscribe_camera(camera_name, camera_topic)

        for topic, label in topics.joint_states:
            self.create_subscription(
                JointState,
                topic,
                self._make_joint_states_cb(label),
                qos_profile_sensor_data,
                callback_group=self._cg_joints,
            )

        for topic, label in topics.forward_commands:
            qos_cmd = QoSProfile(depth=10)
            self.create_subscription(
                Float64MultiArray,
                topic,
                self._make_forward_commands_cb(label),
                qos_cmd,
                callback_group=self._cg_cmd,
            )

        self.get_logger().info("Rerun bridge started.")
        self.get_logger().info(f"State clear gap threshold: {clear_state_gap_s:.3f}s")

    def _subscribe_camera(self, camera_name: str, camera_topic: str) -> None:
        path = f"cameras/{camera_name}"

        def on_compressed(msg: CompressedImage) -> None:
            rr.set_time("ros_time", timestamp=stamp_to_datetime64(msg.header.stamp))
            mt = media_type_from_compressed_format(msg.format) or "image/jpeg"
            rr.log(path, rr.EncodedImage(contents=bytes(msg.data), media_type=mt))

        def on_raw(img: Image) -> None:
            rr.set_time("ros_time", timestamp=stamp_to_datetime64(img.header.stamp))
            rr.log(path, rr.Image(rgb8_to_numpy(img), color_model="RGB"))

        if camera_topic.endswith("/compressed"):
            self.create_subscription(
                CompressedImage,
                camera_topic,
                on_compressed,
                qos_profile_sensor_data,
                callback_group=self._cg_images,
            )
        else:
            self.create_subscription(
                Image,
                camera_topic,
                on_raw,
                qos_profile_sensor_data,
                callback_group=self._cg_images,
            )

    def _next_action_time(self) -> tuple[np.datetime64, np.datetime64] | None:
        with self._time_lock:
            if self._last_ros_time is None:
                return None
            ts = self._last_ros_time
            prev_action_ts = self._last_action_time
            if ts <= prev_action_ts:
                ts = prev_action_ts + np.timedelta64(1, "ns")
            self._last_action_time = ts
            return ts, prev_action_ts

    def _make_joint_states_cb(self, label: str):
        def on_joint_states(msg: JointState) -> None:
            ts = stamp_to_datetime64(msg.header.stamp)
            with self._time_lock:
                self._last_ros_time = ts
            rr.set_time("ros_time", timestamp=ts)
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    log_scalar(f"state/position/{label}{name}", float(msg.position[i]))

        return on_joint_states

    def _make_forward_commands_cb(self, label: str):
        def on_forward_commands(msg: Float64MultiArray) -> None:
            # Float64MultiArray has no header stamp. Derive time from the latest
            # stamped ROS message so action/state plots stay aligned.
            action_time = self._next_action_time()
            if action_time is None:
                return
            ts, prev_action_ts = action_time
            rr.set_time("ros_time", timestamp=ts)

            if (
                self._clear_state_gap > np.timedelta64(0, "ns")
                and ts - prev_action_ts > self._clear_state_gap
            ):
                self.get_logger().info(
                    "Clearing state/position after command gap of %.3fs"
                    % float((ts - prev_action_ts) / np.timedelta64(1, "ms")) / 1000.0
                )
                rr.log("action/position", rr.Clear(recursive=True))
                rr.log("state/position", rr.Clear(recursive=True))

            data = list(msg.data)
            if not self._cmd_joint_order:
                # If you didn't pass joint names, log by index.
                for i, v in enumerate(data):
                    log_scalar(f"action/forward_commands/{label}idx_{i}", float(v))
                return

            # Controller expects commands in the same order as its configured "joints" list.
            n = min(len(self._cmd_joint_order), len(data))
            for i in range(n):
                jn = self._cmd_joint_order[i]
                log_scalar(f"action/position/{label}{jn}", float(data[i]))

        return on_forward_commands


def _arm_label(topic: str) -> str:
    """Dataset-label prefix for one arm topic ('left.' for /follower_left/...)."""
    namespace = topic.strip("/").split("/")[0]
    return follower_label(namespace)


def main() -> None:
    p = argparse.ArgumentParser(description="SO-101 ROS2 to Rerun bridge")
    p.add_argument(
        "--setup",
        required=True,
        choices=tuple(SETUP_CAMERA_NAMES),
        help="Required canonical setup",
    )
    p.add_argument(
        "--joint-states",
        nargs="+",
        default=None,
        help="Follower joint-state topics (default: derived from the setup)",
    )
    p.add_argument(
        "--forward-commands",
        nargs="+",
        default=None,
        help="Follower command topics (default: derived from the setup)",
    )
    p.add_argument(
        "--cmd-joints",
        nargs="*",
        default=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        help="Joint name order matching controller 'joints' param",
    )
    p.add_argument(
        "--clear-state-gap-s",
        type=float,
        default=2.0,
        help="Clear state/position when commands resume after this many seconds; <=0 disables.",
    )
    p.add_argument(
        "--viewer",
        choices=["native", "web"],
        default="web",
        help="Launch web viewer (default) or native desktop app.",
    )
    p.add_argument(
        "--rerun-memory-limit",
        default="512MiB",
        help="Rerun gRPC server memory limit, e.g. 128MiB, 256MiB, 1GiB.",
    )

    args, unknownargs = p.parse_known_args()

    # Initialise Rerun recording.
    rr.init("so101_ros2_live")

    if args.viewer == "native":
        rr.spawn()
    else:
        server_uri = rr.serve_grpc(server_memory_limit=args.rerun_memory_limit)
        rr.serve_web_viewer(connect_to=server_uri)

    # ──  # Blueprint: cameras left, plots right (state + action)
    camera_views = [
        rrb.Spatial2DView(
            name=" ".join(part.capitalize() for part in camera_id.split("_")) + " Camera",
            origin=f"cameras/{camera_id}",
        )
        for camera_id in image_topics(args.setup, compressed=True)
    ]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                *camera_views,
                row_shares=[1] * len(camera_views),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="State (Joint Positions)", origin="state/position"
                ),
                rrb.TimeSeriesView(name="Action (Commands)", origin="action"),
                row_shares=[1, 1],
            ),
            column_shares=[1, 1],
        ),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(blueprint)

    rclpy.init(args=unknownargs)
    joints = args.joint_states or list(joint_state_topics(args.setup))
    commands = args.forward_commands or list(command_topics(args.setup))
    topics = Topics(
        cameras=image_topics(args.setup, compressed=True),
        joint_states=[(topic, _arm_label(topic)) for topic in joints],
        forward_commands=[(topic, _arm_label(topic)) for topic in commands],
    )

    node = So101Ros2ToRerun(
        topics,
        cmd_joint_order=args.cmd_joints,
        clear_state_gap_s=args.clear_state_gap_s,
    )

    exec_ = MultiThreadedExecutor()
    exec_.add_node(node)
    try:
        exec_.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec_.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
