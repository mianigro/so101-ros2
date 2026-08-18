#!/usr/bin/env python3
"""SO-101 Episode Browser — Rerun RecordingStream + ROS2 bag playback + Gradio UI."""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import numpy as np
import rclpy
import rerun as rr
from gradio_rerun import Rerun
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray

from so101_setups import (
    SETUP_CAMERA_NAMES,
    command_topics,
    detect_recorded_setup,
    follower_label,
    image_topics,
    joint_state_topics,
)

# ---------------------------------------------------------------------------
# Constants & Styling
# ---------------------------------------------------------------------------

APP_ID = "so101_episode_browser"

CSS = """
#episode_list_wrap {
  height: 750px;
  overflow-y: auto;
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 8px;
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def media_type_from_compressed_format(fmt: str) -> str | None:
    f = (fmt or "").lower()
    if "jpeg" in f or "jpg" in f:
        return "image/jpeg"
    if "png" in f:
        return "image/png"
    return None


def rgb8_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)


def stamp_to_datetime64(stamp) -> np.datetime64:
    return np.datetime64(stamp.sec * 1_000_000_000 + stamp.nanosec, "ns")



# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Topics:
    cameras: dict[str, str] = field(default_factory=dict)
    # (topic, dataset label prefix) per follower arm, in setup order
    joint_states: list[tuple[str, str]] = field(default_factory=list)
    forward_commands: list[tuple[str, str]] = field(default_factory=list)


def _arm_label(topic: str) -> str:
    """Dataset-label prefix for one arm topic ('left.' for /follower_left/...)."""
    return follower_label(topic.strip("/").split("/")[0])


DEFAULT_CMD_JOINTS: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ---------------------------------------------------------------------------
# Rerun blueprint
# ---------------------------------------------------------------------------


def make_blueprint(camera_ids: list[str]) -> rr.blueprint.Blueprint:
    camera_views = [
        rr.blueprint.Spatial2DView(
            name=" ".join(part.capitalize() for part in camera_id.split("_")),
            origin=f"cameras/{camera_id}",
        )
        for camera_id in camera_ids
    ]
    return rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Vertical(
                *camera_views,
            ),
            rr.blueprint.Vertical(
                rr.blueprint.TimeSeriesView(name="Joint States", origin="state"),
                rr.blueprint.TimeSeriesView(name="Actions", origin="action"),
            ),
        ),
        collapse_panels=True,
    )


# ---------------------------------------------------------------------------
# ROS2 Node — So101Ros2ToRerun (stream-aware)
# ---------------------------------------------------------------------------


class So101Ros2ToRerun(Node):
    """Subscribe to SO-101 topics and log data to a swappable RecordingStream."""

    def __init__(self, topics: Topics, cmd_joint_order: list[str]) -> None:
        super().__init__("so101_ros2_to_rerun")
        self._cmd_joint_order = list(cmd_joint_order)
        self.camera_ids = list(topics.cameras)

        # -- Swappable RecordingStream --
        self._rec: rr.RecordingStream | None = None

        # Timestamp reference for unstamped messages (Float64MultiArray).
        # Stamped callbacks write here; the action callback reads it.
        # Protected by _time_lock for thread-safety.
        self._time_lock = threading.Lock()
        self._last_ros_time: np.datetime64 | None = None
        self._last_action_time: np.datetime64 | None = None

        # -- Callback groups --
        self._cg_images = ReentrantCallbackGroup()
        self._cg_joints = ReentrantCallbackGroup()
        self._cg_cmd = ReentrantCallbackGroup()

        # -- Subscriptions --
        for camera_id, camera_topic in topics.cameras.items():
            self._subscribe_camera(camera_id, camera_topic)

        for topic, label in topics.joint_states:
            self.create_subscription(
                JointState, topic,
                partial(self._on_joint_states, label=label),
                qos_profile_sensor_data, callback_group=self._cg_joints,
            )

        for topic, label in topics.forward_commands:
            qos_cmd = QoSProfile(depth=10)
            self.create_subscription(
                Float64MultiArray, topic,
                partial(self._on_forward_commands, label=label),
                qos_cmd, callback_group=self._cg_cmd,
            )

        self.get_logger().info("Rerun bridge started (stream-aware).")

    # -- public API --

    def set_recording(self, rec: rr.RecordingStream | None) -> None:
        """Hot-swap the target RecordingStream."""
        self._rec = rec
        with self._time_lock:
            self._last_ros_time = None
            self._last_action_time = None

    def _get_rec(self) -> rr.RecordingStream | None:
        return self._rec

    # -- helpers --

    @staticmethod
    def _is_compressed(topic_name: str) -> bool:
        return "compressed" in topic_name.lower()

    # -- timestamp helpers --

    def _next_action_time(self) -> np.datetime64 | None:
        """Return a strictly-increasing timestamp for unstamped action msgs.

        Derives from the last stamped reference time. Tracks its own
        monotonic counter so consecutive actions never share a timestamp,
        even if the stamped reference hasn't changed.

        Returns None when no stamped reference exists yet.
        """
        with self._time_lock:
            if self._last_ros_time is None:
                return None
            ts = self._last_ros_time
            prev = self._last_action_time
            if prev is not None and ts <= prev:
                ts = prev + np.timedelta64(1, "ns")
            self._last_action_time = ts
            return ts

    # -- image callbacks --

    def _subscribe_camera(self, camera_id: str, camera_topic: str) -> None:
        path = f"cameras/{camera_id}"

        def on_compressed(msg: CompressedImage) -> None:
            rec = self._get_rec()
            if rec is None:
                return
            mt = media_type_from_compressed_format(msg.format) or "image/jpeg"
            rec.set_time("ros_time", timestamp=stamp_to_datetime64(msg.header.stamp))
            rec.log(path, rr.EncodedImage(contents=bytes(msg.data), media_type=mt))

        def on_raw(msg: Image) -> None:
            rec = self._get_rec()
            if rec is None:
                return
            rec.set_time("ros_time", timestamp=stamp_to_datetime64(msg.header.stamp))
            rec.log(path, rr.Image(rgb8_to_numpy(msg), color_model="RGB"))

        if self._is_compressed(camera_topic):
            self.create_subscription(
                CompressedImage, camera_topic, on_compressed,
                qos_profile_sensor_data, callback_group=self._cg_images,
            )
        else:
            self.create_subscription(
                Image, camera_topic, on_raw,
                qos_profile_sensor_data, callback_group=self._cg_images,
            )

    # -- joint state callback --

    def _on_joint_states(self, msg: JointState, label: str = "") -> None:
        rec = self._get_rec()
        if rec is None:
            return
        ts = stamp_to_datetime64(msg.header.stamp)
        # Cache as the reference clock for unstamped action messages.
        with self._time_lock:
            self._last_ros_time = ts
        rec.set_time("ros_time", timestamp=ts)
        for name, pos in zip(msg.name, msg.position):
            rec.log(f"state/position/{label}{name}", rr.Scalars(float(pos)))

    # -- forward commands callback --

    def _on_forward_commands(self, msg: Float64MultiArray, label: str = "") -> None:
        rec = self._get_rec()
        if rec is None:
            return
        # Float64MultiArray has no header — derive a monotonic timestamp
        # from the last known stamped time.
        ts = self._next_action_time()
        if ts is None:
            return  # no stamped reference yet, skip
        rec.set_time("ros_time", timestamp=ts)
        for name, val in zip(self._cmd_joint_order, msg.data):
            rec.log(f"action/position/{label}{name}", rr.Scalars(float(val)))


# ---------------------------------------------------------------------------
# Episode indexing
# ---------------------------------------------------------------------------


def index_episodes(root: Path, setup: str) -> list[tuple[str, str]]:
    """Find all *.mcap files under *root*, return [(label, path_str), ...]."""
    mcaps = sorted(root.rglob("*.mcap"))
    episodes: list[tuple[str, str]] = []
    for mcap in mcaps:
        detected = detect_recorded_setup(mcap.parent)
        if detected != setup:
            raise SystemExit(
                f"{mcap.parent}: selected setup {setup!r}, "
                f"but cameras match {detected!r}"
            )
        episodes.append((f"{mcap.parent.name}/{mcap.name}", str(mcap)))
    return episodes


# ---------------------------------------------------------------------------
# Bag playback management
# ---------------------------------------------------------------------------

BAG_PROC: subprocess.Popen | None = None
BAG_LOCK = threading.Lock()


def stop_playback() -> None:
    global BAG_PROC
    with BAG_LOCK:
        if BAG_PROC is not None:
            try:
                BAG_PROC.send_signal(signal.SIGINT)
                BAG_PROC.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    BAG_PROC.kill()
                    BAG_PROC.wait(timeout=2)
                except OSError:
                    pass
            BAG_PROC = None


def start_playback(
    target: str,
    rate: float = 1.0,
    loop: bool = False,
    read_ahead: int = 12000,
) -> subprocess.Popen:
    global BAG_PROC
    stop_playback()

    target_path = Path(target)
    if target_path.is_file():
        bag_dir = str(target_path.parent)
    else:
        bag_dir = str(target_path)

    cmd = [
        "ros2", "bag", "play",
        "-s", "mcap",
        bag_dir,
        "--rate", str(float(rate)),
        "--read-ahead-queue-size", str(int(read_ahead)),
        "--disable-keyboard-controls",
    ]
    if loop:
        cmd.append("--loop")

    with BAG_LOCK:
        BAG_PROC = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return BAG_PROC


# ---------------------------------------------------------------------------
# Streaming logic (mirrors MCAP viewer pattern)
# ---------------------------------------------------------------------------


def stream_episode(
    episode_label: str,
    episodes_state: dict[str, str],
    node: So101Ros2ToRerun,
    current_recording_id: str,
) -> Iterator[tuple[Any, Any, str]]:
    """Generator that creates a new RecordingStream per episode, starts bag
    playback, and yields binary chunks to the Gradio Rerun component."""

    if not episode_label:
        yield gr.skip(), "⚠️ No episode selected.", current_recording_id
        return

    target = episodes_state.get(episode_label, "")
    if not target:
        yield gr.skip(), f"⚠️ Episode not found: {episode_label}", current_recording_id
        return

    # -- Create a fresh RecordingStream --
    new_recording_id = str(uuid.uuid4())
    rec = rr.RecordingStream(application_id=APP_ID, recording_id=new_recording_id)
    stream = rec.binary_stream()

    # Send blueprint on this stream
    rec.send_blueprint(make_blueprint(node.camera_ids))

    # Hot-swap the recording on the ROS2 node
    node.set_recording(rec)

    # Start bag playback (publishes to ROS2 topics → node callbacks → rec)
    stop_playback()
    proc = start_playback(target, rate=1.0, loop=False, read_ahead=12000)

    yield gr.skip(), f"▶️ Loading `{episode_label}`...", new_recording_id

    try:
        while proc.poll() is None:
            chunk = stream.read()
            if chunk:
                yield chunk, gr.skip(), new_recording_id
            else:
                time.sleep(0.01)

        # Flush remaining data after playback ends
        while True:
            chunk = stream.read()
            if not chunk:
                break
            yield chunk, gr.skip(), new_recording_id

        yield gr.skip(), f"✅ `{episode_label}` complete", new_recording_id

    except GeneratorExit:
        stop_playback()
    finally:
        node.set_recording(None)
        rr.disconnect(recording=rec)


# ---------------------------------------------------------------------------
# Gradio UI builder
# ---------------------------------------------------------------------------


def build_ui(
    episodes: list[tuple[str, str]],
    node: So101Ros2ToRerun,
) -> gr.Blocks:
    """Build the Gradio Blocks interface with embedded Rerun streaming viewer."""

    episode_labels = [label for label, _ in episodes]
    episode_map = {label: path for label, path in episodes}

    with gr.Blocks(title="SO-101 Episode Browser", theme=gr.themes.Soft(), css=CSS) as demo:

        # --- State ---
        episodes_state = gr.State(episode_map)
        recording_id = gr.State("")

        # --- Layout ---
        gr.Markdown("# 🤖 SO-101 Episode Browser")

        with gr.Row():
            # -- Left column: controls --
            with gr.Column(scale=1, min_width=300):
                episode_radio = gr.Radio(
                    choices=episode_labels,
                    label="Episodes",
                    info=f"{len(episodes)} episode(s) found",
                )

                status_md = gr.Markdown("Ready.")

            # -- Right column: Rerun streaming viewer --
            with gr.Column(scale=3):
                viewer = Rerun(
                    streaming=True,
                    height=800,
                    panel_states={
                        "blueprint": "hidden",
                        "selection": "hidden",
                        "time": "collapsed",
                    },
                )

        # -- Events --
        # We need to pass the node to stream_episode; wrap in a closure
        def _stream(episode_label, ep_state, rec_id):
            yield from stream_episode(episode_label, ep_state, node, rec_id)

        episode_radio.change(
            fn=_stream,
            inputs=[episode_radio, episodes_state, recording_id],
            outputs=[viewer, status_md, recording_id],
            concurrency_limit=1,
        )

    return demo


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SO-101 Episode Browser — Rerun RecordingStream + ROS2 bag + Gradio",
    )
    p.add_argument(
        "--episodes_root", type=str, required=True,
        help="Root directory containing MCAP episode files",
    )
    p.add_argument(
        "--setup",
        required=True,
        choices=tuple(SETUP_CAMERA_NAMES),
        help="Required canonical setup",
    )
    p.add_argument(
        "--joint-states", type=str, nargs="+", default=None,
        help="Follower joint-state topics (default: derived from the setup)",
    )
    p.add_argument(
        "--forward-commands", type=str, nargs="+", default=None,
        help="Forward command topics; set empty to disable (default: derived from the setup)",
    )
    p.add_argument(
        "--cmd-joints", type=str, nargs="+",
        default=DEFAULT_CMD_JOINTS,
        help="Joint names for forward commands (order matters)",
    )
    p.add_argument("--server_name", type=str, default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=7860)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    if not episodes_root.is_dir():
        print(f"Error: episodes_root does not exist: {episodes_root}", file=sys.stderr)
        sys.exit(1)

    # -- Index episodes --
    episodes = index_episodes(episodes_root, args.setup)
    print(f"Found {len(episodes)} MCAP episode(s) under {episodes_root}")
    for label, _ in episodes:
        print(f"  • {label}")

    # -- Init ROS2 --
    rclpy.init()
    joints = args.joint_states or list(joint_state_topics(args.setup))
    commands = args.forward_commands or list(command_topics(args.setup))
    topics = Topics(
        cameras=image_topics(args.setup, compressed=True),
        joint_states=[(topic, _arm_label(topic)) for topic in joints],
        forward_commands=[(topic, _arm_label(topic)) for topic in commands],
    )
    node = So101Ros2ToRerun(topics, args.cmd_joints)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    print("ROS2 bridge node running in background thread.")

    # -- Cleanup --
    def cleanup() -> None:
        print("\nShutting down...")
        stop_playback()
        try:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    atexit.register(cleanup)

    # -- Build and launch Gradio --
    demo = build_ui(episodes, node)
    print(f"Launching Gradio on {args.server_name}:{args.server_port}")
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        inbrowser=True,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
