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

"""Bridge ROS 2 node: mission dispatch packages -> demo-conditioned rollouts.

Consumes the external ``ros2_action_goal:v1`` dispatch-package contract
(``GET /api/mission/<ws>/<id>/dispatch-package``) as-is — no upstream Bridge
Robot changes. The core logic (selection, ordering, state machine, asset
resolution) is ROS-free and unit-tested against canned packages; the rclpy
shell below is a thin timer + parameter wrapper.

Per subtask: top-k demo selection (success outcomes first, then recency
``prov_id``, then ranker score), keyframe fetch -> decode -> the SHARED
policy image transform, trajectory slice when available, then
``set_demo_pack`` on the side channel. On subtask switch: ``clear`` then
re-encode. Fallback mode ``conditioning: prompt`` (Path 1: language
enrichment only, no demo tokens) produces the baseline arm of every
comparison — same node, config flag.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import TrajNormalizer, preprocess_demo_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Dispatch package model (ros2_action_goal:v1 subset this node consumes)  #
# ---------------------------------------------------------------------- #


@dataclass
class DemoAsset:
    demonstration: dict  # raw dispatch entry (episode_id, keyframes, outcome, ...)
    frames: np.ndarray | None = None       # [F, 3, 224, 224] preprocessed, or None until resolved
    traj: np.ndarray | None = None         # [S, d] normalized, or None
    traj_ok: float = 0.0


@dataclass
class Subtask:
    subtask_id: str
    depends_on: list[str]
    prompt: str
    conditioning_mode: str = "demo"        # "demo" | "prompt"
    demonstrations: list[dict] = field(default_factory=list)
    terminal_topic: str | None = None

    @property
    def is_terminal_predicate(self) -> bool:
        return self.terminal_topic is not None


def parse_dispatch_package(package: dict) -> list[Subtask]:
    """Extract the ordered-subtask view of a dispatch package."""
    subtasks = []
    for st in package.get("subtasks", []):
        cond = st.get("conditioning", {}) or {}
        subtasks.append(
            Subtask(
                subtask_id=st["id"],
                depends_on=list(st.get("depends_on", [])),
                prompt=str(st.get("action_spec", {}).get("prompt", st.get("prompt", ""))),
                conditioning_mode=cond.get("mode", "prompt"),
                demonstrations=list(cond.get("demonstrations", [])),
                terminal_topic=st.get("action_spec", {}).get("terminal", {}).get("topic"),
            )
        )
    return subtasks


def order_subtasks(subtasks: list[Subtask]) -> list[Subtask]:
    """Topological order over depends_on edges (stable for ties)."""
    by_id = {s.subtask_id: s for s in subtasks}
    done: set[str] = set()
    ordered: list[Subtask] = []
    remaining = list(subtasks)
    while remaining:
        ready = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not ready:
            raise ValueError(f"cycle or unknown dependency among {[s.subtask_id for s in remaining]}")
        for s in ready:
            ordered.append(s)
            done.add(s.subtask_id)
            remaining.remove(s)
    assert len(ordered) == len(by_id)
    return ordered


def select_top_k(demonstrations: list[dict], k: int) -> list[dict]:
    """Rank: success outcomes first, then recency (prov_id), then score."""
    def rank_key(d: dict):
        outcome = str(d.get("outcome", "")).lower()
        success = 0 if outcome in ("success", "succeeded", "ok") else 1
        return (success, -int(d.get("prov_id", 0)), -float(d.get("ranker_score", 0.0)))

    return sorted(demonstrations, key=rank_key)[:k]


# ---------------------------------------------------------------------- #
# Asset resolution                                                        #
# ---------------------------------------------------------------------- #


def fetch_keyframes(demo: dict, frames_per_demo: int) -> np.ndarray | None:
    """Presigned keyframe URLs -> decoded -> SHARED policy transform.

    Returns ``[F, 3, 224, 224]`` float32 in [-1, 1] or None when the demo has
    no usable keyframes. Local ``file://`` and http(s):// URLs both work
    (tests use file://).
    """
    import cv2

    urls = demo.get("keyframes", []) or []
    if not urls:
        return None
    # Resample to exactly frames_per_demo regardless of how many keyframes
    # the dispatch package carries (too few get repeated, start/goal kept).
    idx = np.linspace(0, len(urls) - 1, frames_per_demo).round().astype(int)
    urls = [urls[i] for i in idx]
    frames = []
    for url in urls:
        url = str(url)
        if url.startswith("file://"):
            data = Path(url[7:]).read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (presigned URLs)
                data = resp.read()
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    stacked = torch.stack(frames)  # [F, 3, H, W] in [0, 1]
    return preprocess_demo_frames(stacked).numpy()


def resolve_demo(demo: dict, normalizer: TrajNormalizer | None, frames_per_demo: int) -> DemoAsset:
    asset = DemoAsset(demonstration=demo)
    asset.frames = fetch_keyframes(demo, frames_per_demo)
    trajectory = demo.get("trajectory", {}) or {}
    if normalizer is not None and trajectory.get("available") and trajectory.get("states") is not None:
        states = torch.as_tensor(np.asarray(trajectory["states"], dtype=np.float32))
        actions = torch.as_tensor(
            np.asarray(trajectory.get("actions", np.zeros_like(states)), dtype=np.float32)
        )
        s_idx = np.linspace(0, max(1, states.shape[0] - 1), 16).round().astype(int)
        asset.traj = normalizer(states[s_idx], actions[s_idx]).numpy()
        asset.traj_ok = 1.0
    return asset


# ---------------------------------------------------------------------- #
# State machine                                                           #
# ---------------------------------------------------------------------- #


class BridgeState:
    """One active subtask; only its conditioning pack is in context."""

    def __init__(self, subtasks: list[Subtask]):
        self.subtasks = order_subtasks(subtasks)
        self.active_index = -1
        self.completed: list[str] = []

    @property
    def active(self) -> Subtask | None:
        if 0 <= self.active_index < len(self.subtasks):
            return self.subtasks[self.active_index]
        return None

    def advance(self) -> Subtask | None:
        if self.active is not None:
            self.completed.append(self.active.subtask_id)
        self.active_index += 1
        return self.active

    @property
    def finished(self) -> bool:
        return self.active_index >= len(self.subtasks)


class TerminalMonitor:
    """Gate subtask advancement on fired terminal events (ROS-free core).

    A subtask with a terminal predicate (``action_spec.terminal.topic``)
    advances only after :meth:`record_event` sees that exact topic; a
    subtask without one advances as soon as its pack is pushed. The rclpy
    wrapper and the polling CLI both drive the same monitor.
    """

    def __init__(self, state: BridgeState):
        self.state = state
        self._fired: set[str] = set()
        self._active_topic: str | None = None

    def set_active(self, subtask: Subtask | None) -> None:
        self._active_topic = subtask.terminal_topic if subtask else None

    def record_event(self, topic: str) -> None:
        """A terminal event fired on ``topic`` (idempotent)."""
        self._fired.add(topic)

    def should_advance(self) -> bool:
        active = self.state.active
        if active is None:
            return False
        if active.terminal_topic is None:
            return True  # no predicate: the pack push IS the transition
        return active.terminal_topic in self._fired


# ---------------------------------------------------------------------- #
# rclpy shell (import-time optional)                                      #
# ---------------------------------------------------------------------- #


def build_demo_pack(
    active: Subtask,
    transport,
    *,
    k: int,
    frames_per_demo: int,
    k_max: int,
    normalizer: TrajNormalizer | None = None,
) -> dict:
    """Select, resolve and push the active subtask's demo pack.

    Returns the transport reply ({"status": "ok", "encode_s": ...}) or a
    {"status": "skipped"} marker for prompt-conditioned subtasks (the
    baseline arm clears any previous pack).
    """
    if active.conditioning_mode != "demo" or not active.demonstrations:
        transport.clear()
        return {"status": "skipped", "reason": f"conditioning={active.conditioning_mode}"}

    selected = select_top_k(active.demonstrations, k)
    assets = [resolve_demo(d, normalizer, frames_per_demo) for d in selected]
    assets = [a for a in assets if a.frames is not None][:k_max]
    if not assets:
        transport.clear()
        return {"status": "skipped", "reason": "no resolvable keyframes"}

    frames = np.stack([a.frames for a in assets])          # [k, F, 3, 224, 224]
    # Per-asset stacking: a demo without a trajectory contributes zero rows
    # with traj_ok=0 (the DemoEncoder masks that slot's traj branch) — never
    # drop the trajectories of the OTHER selected demos.
    traj = np.stack(
        [a.traj if a.traj is not None else np.zeros((16, 64), np.float32) for a in assets]
    )
    traj_ok = np.asarray([a.traj_ok for a in assets], dtype=np.float32)
    return transport.set_demo_pack(frames, traj=traj, traj_ok=traj_ok, k_max=k_max)


def run_mission(
    state: BridgeState,
    transport,
    *,
    k: int,
    frames_per_demo: int,
    k_max: int = 4,
    normalizer: TrajNormalizer | None = None,
    advance_mode: str = "terminal",
    should_advance=None,
    clear_on_advance=True,
) -> list[str]:
    """Drive one dispatch mission: pack per active subtask, wait, advance.

    ``advance_mode``: ``"terminal"`` gates each subtask on
    ``should_advance()`` (the :class:`TerminalMonitor` verdict, or a custom
    callable); ``"auto"`` advances immediately after each pack push (legacy
    smoke behavior). Returns the completed subtask ids in order.
    """
    monitor = TerminalMonitor(state)
    while not state.finished:
        active = state.advance()
        if active is None:
            break
        monitor.set_active(active)
        reply = build_demo_pack(
            active, transport, k=k, frames_per_demo=frames_per_demo,
            k_max=k_max, normalizer=normalizer,
        )
        logger.info("subtask %s: %s", active.subtask_id, reply)
        if advance_mode == "terminal":
            gate = should_advance or monitor.should_advance
            while not gate():
                time.sleep(0.05)
        if clear_on_advance:
            transport.clear()
    return state.completed


def main(argv=None) -> int:
    """ROS-free CLI entry: poll dispatch, drive the pack per active subtask.

    With ``--advance-mode terminal`` the driver waits for terminal events
    recorded on stdin (one topic name per line) — the test/smoke harness.
    Live ROS operation uses :class:`BridgeICLNode` below.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ICL bridge node (dispatch -> demo packs)")
    parser.add_argument("--api-base", required=True, help="e.g. https://bridge.internal/api")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mission-id", required=True)
    parser.add_argument("--demo-host", default="127.0.0.1")
    parser.add_argument("--demo-port", type=int, default=8661)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--frames-per-demo", type=int, default=6)
    parser.add_argument("--stats", default=None, help="stage_stats.json for traj normalization")
    parser.add_argument("--poll-s", type=float, default=5.0, help="dispatch poll period")
    parser.add_argument("--advance-mode", default="terminal", choices=["terminal", "auto"],
                        help="terminal: wait for terminal events; auto: advance immediately")
    parser.add_argument("--stdin-events", action="store_true",
                        help="terminal mode reads topic names from stdin, one per line")
    parser.add_argument("--once", action="store_true", help="process one dispatch package and exit")
    args = parser.parse_args(argv)

    from .data import TrajNormalizer as _T
    from .demo_transport import DemoTransportClient, load_stage_stats

    normalizer = None
    if args.stats:
        normalizer = _T(load_stage_stats(args.stats), d_state=6, d_action=6)
    transport = DemoTransportClient(args.demo_host, args.demo_port)

    should_advance = None
    monitor_holder = None
    if args.advance_mode == "terminal" and args.stdin_events:
        monitor_holder = {}
        import threading

        def _reader():
            for line in sys.stdin:
                topic = line.strip()
                if topic and monitor_holder.get("monitor"):
                    monitor_holder["monitor"].record_event(topic)

        threading.Thread(target=_reader, daemon=True).start()
        should_advance = lambda: monitor_holder["monitor"].should_advance()  # noqa: E731

    url = f"{args.api_base}/mission/{args.workspace}/{args.mission_id}/dispatch-package"
    while True:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (internal API)
            package = json.loads(resp.read())
        state = BridgeState(parse_dispatch_package(package))
        if monitor_holder is not None:
            monitor_holder["monitor"] = TerminalMonitor(state)
        completed = run_mission(
            state, transport, k=args.k, frames_per_demo=args.frames_per_demo,
            k_max=4, normalizer=normalizer, advance_mode=args.advance_mode,
            should_advance=should_advance,
        )
        logger.info("mission %s completed subtasks: %s", args.mission_id, completed)
        if args.once:
            return 0
        time.sleep(args.poll_s)


try:
    import rclpy  # noqa: F401
    from rclpy.node import Node as _RclpyNode

    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False


if _RCLPY_AVAILABLE:

    class BridgeICLNode(_RclpyNode):
        """ROS 2 shell: dispatch polling + terminal-topic subscriptions.

        The core logic (ordering, selection, state machine, monitor) lives in
        the ROS-free classes above and is unit-tested; this shell only wires
        parameters, the dispatch HTTP poll, dynamic per-subtask subscriptions
        (``--terminal-msg-type``, default ``std_msgs/Empty``) and the demo
        transport. Not yet exercised against a live Bridge Robot.
        """

        def __init__(self):
            super().__init__("bridge_icl_node")
            self.declare_parameter("api_base", "")
            self.declare_parameter("workspace", "")
            self.declare_parameter("mission_id", "")
            self.declare_parameter("demo_host", "127.0.0.1")
            self.declare_parameter("demo_port", 8661)
            self.declare_parameter("k", 4)
            self.declare_parameter("frames_per_demo", 6)
            self.declare_parameter("stats_path", "")
            self.declare_parameter("poll_period_s", 5.0)
            self.declare_parameter("terminal_msg_type", "std_msgs/msg/Empty")

            from .demo_transport import DemoTransportClient, load_stage_stats

            stats_path = self.get_parameter("stats_path").value
            self.normalizer = None
            if stats_path:
                from .data import TrajNormalizer

                self.normalizer = TrajNormalizer(load_stage_stats(stats_path), d_state=6, d_action=6)
            self.transport = DemoTransportClient(
                self.get_parameter("demo_host").value,
                int(self.get_parameter("demo_port").value),
            )
            self.state: BridgeState | None = None
            self.monitor: TerminalMonitor | None = None
            self._subscription = None
            self.poll_timer = self.create_timer(
                float(self.get_parameter("poll_period_s").value), self._poll_dispatch
            )

        def _poll_dispatch(self) -> None:
            if self.state is None:
                import urllib.request

                base = self.get_parameter("api_base").value
                ws = self.get_parameter("workspace").value
                mission = self.get_parameter("mission_id").value
                url = f"{base}/mission/{ws}/{mission}/dispatch-package"
                with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                    package = json.loads(resp.read())
                self.state = BridgeState(parse_dispatch_package(package))
                self.monitor = TerminalMonitor(self.state)
                self._advance_and_push()

        def _advance_and_push(self) -> None:
            active = self.state.advance()
            if active is None:
                self.get_logger().info("mission complete")
                self.state = None
                self._drop_subscription()
                return
            self.monitor.set_active(active)
            reply = build_demo_pack(
                active, self.transport,
                k=int(self.get_parameter("k").value),
                frames_per_demo=int(self.get_parameter("frames_per_demo").value),
                k_max=4, normalizer=self.normalizer,
            )
            self.get_logger().info(f"subtask {active.subtask_id}: {reply}")
            self._subscribe_terminal(active)

        def _subscribe_terminal(self, active: Subtask) -> None:
            self._drop_subscription()
            if active.terminal_topic is None:
                return
            msg_type = self.get_parameter("terminal_msg_type").value
            module, _, cls = msg_type.replace("/msg/", "/").rpartition("/")
            import importlib

            mod = importlib.import_module(f"{module.replace('/', '.')}.msg")
            self._subscription = self.create_subscription(
                getattr(mod, cls), active.terminal_topic,
                lambda _msg: self._on_terminal(active.terminal_topic), 1,
            )

        def _drop_subscription(self) -> None:
            if self._subscription is not None:
                self.destroy_subscription(self._subscription)
                self._subscription = None

        def _on_terminal(self, topic: str) -> None:
            self.monitor.record_event(topic)
            if self.monitor.should_advance():
                self.transport.clear()
                self._advance_and_push()


else:  # pragma: no cover - exercised only outside ROS environments
    class BridgeICLNode:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "rclpy is not available in this environment; run the ROS-free "
                "CLI (so101_icl.bridge_icl_node:main) or use a ROS 2 shell."
            )


if __name__ == "__main__":
    raise SystemExit(main())
