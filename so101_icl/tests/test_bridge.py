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

"""Bridge node unit tests against canned dispatch packages (no ROS, no GPU)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101_icl.bridge_icl_node import (  # noqa: E402
    BridgeState,
    TerminalMonitor,
    build_demo_pack,
    order_subtasks,
    parse_dispatch_package,
    run_mission,
    select_top_k,
)

try:
    import torch  # noqa: F401

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def _package():
    return {
        "mission_id": "m1",
        "subtasks": [
            {
                "id": "st_open",
                "depends_on": [],
                "action_spec": {"prompt": "open the drawer",
                                 "terminal": {"topic": "/events/drawer_open"}},
                "conditioning": {"mode": "demo", "demonstrations": [
                    {"episode_id": 3, "outcome": "failure", "prov_id": 30,
                     "keyframes": ["k3"]},
                    {"episode_id": 1, "outcome": "success", "prov_id": 10, "ranker_score": 0.2,
                     "keyframes": ["k1"]},
                    {"episode_id": 2, "outcome": "success", "prov_id": 20, "ranker_score": 0.9,
                     "keyframes": ["k2"]},
                ]},
            },
            {
                "id": "st_place",
                "depends_on": ["st_open"],
                "action_spec": {"prompt": "place the cube inside"},
                "conditioning": {"mode": "prompt", "demonstrations": []},
            },
        ],
    }


class TestParseOrderSelect(unittest.TestCase):
    def test_parse(self):
        subtasks = parse_dispatch_package(_package())
        self.assertEqual([s.subtask_id for s in subtasks], ["st_open", "st_place"])
        self.assertEqual(subtasks[0].conditioning_mode, "demo")
        self.assertEqual(len(subtasks[0].demonstrations), 3)
        self.assertEqual(subtasks[1].conditioning_mode, "prompt")
        self.assertEqual(subtasks[1].prompt, "place the cube inside")

    def test_order_respects_dependencies(self):
        subtasks = parse_dispatch_package(_package())
        subtasks.reverse()
        ordered = order_subtasks(subtasks)
        self.assertEqual(ordered[0].subtask_id, "st_open")
        with self.assertRaises(ValueError):
            order_subtasks([
                *subtasks,
                type(subtasks[0])("x", ["y"], "p"),
                type(subtasks[0])("y", ["x"], "p"),
            ])

    def test_select_top_k_success_recency_score(self):
        demos = parse_dispatch_package(_package())[0].demonstrations
        ranked = select_top_k(demos, 2)
        # success-first, then recency (prov_id) among successes
        self.assertEqual([d["episode_id"] for d in ranked], [2, 1])
        self.assertEqual(select_top_k(demos, 1)[0]["episode_id"], 2)

    def test_state_machine(self):
        state = BridgeState(parse_dispatch_package(_package()))
        self.assertFalse(state.finished)
        first = state.advance()
        self.assertEqual(first.subtask_id, "st_open")
        second = state.advance()
        self.assertEqual(second.subtask_id, "st_place")
        self.assertFalse(state.finished)  # st_place still active
        third = state.advance()
        self.assertIsNone(third)          # marks st_place complete
        self.assertTrue(state.finished)
        self.assertEqual(state.completed, ["st_open", "st_place"])


class _StubTransport:
    def __init__(self):
        self.calls = []

    def set_demo_pack(self, frames, traj=None, traj_ok=None, k_max=4):
        self.calls.append(("set", frames.shape, int(frames.shape[0])))
        return {"status": "ok", "encode_s": 0.01, "k": int(frames.shape[0])}

    def clear(self):
        self.calls.append(("clear",))
        return {"status": "ok"}


def _write_keyframe_png(path: Path, seed: int):
    img = (np.random.default_rng(seed).random((480, 640, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@unittest.skipUnless(TORCH_OK, "torch not available")
class TestBuildDemoPack(unittest.TestCase):
    def test_prompt_mode_clears(self):
        subtasks = parse_dispatch_package(_package())
        transport = _StubTransport()
        reply = build_demo_pack(subtasks[1], transport, k=4, frames_per_demo=6, k_max=4)
        self.assertEqual(reply["status"], "skipped")
        self.assertEqual(transport.calls, [("clear",)])

    def test_demo_mode_resolves_keyframes(self):
        subtasks = parse_dispatch_package(_package())
        with tempfile.TemporaryDirectory() as tmp:
            for i in (1, 2, 3):
                _write_keyframe_png(Path(tmp) / f"k{i}.png", seed=i)
            for st in subtasks:
                for d in st.demonstrations:
                    d["keyframes"] = [f"file://{tmp}/{k}.png" for k in d["keyframes"]]
            transport = _StubTransport()
            reply = build_demo_pack(subtasks[0], transport, k=2, frames_per_demo=6, k_max=4)
            self.assertEqual(reply["status"], "ok")
            kind, shape, k = transport.calls[0]
            self.assertEqual(kind, "set")
            self.assertEqual(shape, (2, 6, 3, 224, 224))
            self.assertEqual(k, 2)


class _SequenceTransport(_StubTransport):
    """Stub transport that records the ordered set/clear call sequence."""

    def set_demo_pack(self, frames, traj=None, traj_ok=None, k_max=4):
        self.calls.append(("set", int(frames.shape[0])))
        return {"status": "ok", "k": int(frames.shape[0])}


@unittest.skipUnless(TORCH_OK, "torch not available")
class TestTerminalMonitor(unittest.TestCase):
    def test_predicate_blocks_until_own_topic_fires(self):
        state = BridgeState(parse_dispatch_package(_package()))
        monitor = TerminalMonitor(state)
        active = state.advance()          # st_open, terminal /events/drawer_open
        monitor.set_active(active)
        self.assertFalse(monitor.should_advance())
        monitor.record_event("/events/something_else")
        self.assertFalse(monitor.should_advance())
        monitor.record_event("/events/drawer_open")
        self.assertTrue(monitor.should_advance())

    def test_no_predicate_advances_after_pack(self):
        subtasks = parse_dispatch_package(_package())
        subtasks[0].terminal_topic = None  # force the first subtask predicate-less
        state = BridgeState(subtasks)
        monitor = TerminalMonitor(state)
        monitor.set_active(state.advance())
        self.assertTrue(monitor.should_advance())

    def test_run_mission_terminal_lifecycle(self):
        package = _package()
        with tempfile.TemporaryDirectory() as tmp:
            for i, st in enumerate(package["subtasks"]):
                for d in st.get("conditioning", {}).get("demonstrations", []):
                    _write_keyframe_png(Path(tmp) / f"k{d['episode_id']}.png", seed=i)
                    d["keyframes"] = [f"file://{tmp}/k{d['episode_id']}.png"]
            state = BridgeState(parse_dispatch_package(package))
            monitor = TerminalMonitor(state)

            events: list[str] = []
            # gate: blocks st_open until its terminal fires, then passes;
            # st_place has no predicate -> passes immediately
            def gate():
                active = state.active
                if active is None:
                    return True
                if active.terminal_topic is None:
                    return True
                return active.terminal_topic in events

            import threading

            def fire_later():
                import time as _t

                _t.sleep(0.2)
                events.append("/events/drawer_open")

            transport = _SequenceTransport()
            threading.Thread(target=fire_later, daemon=True).start()
            completed = run_mission(
                state, transport, k=2, frames_per_demo=6, k_max=4,
                advance_mode="terminal", should_advance=gate,
            )
            self.assertEqual(completed, ["st_open", "st_place"])
            # st_open (demo): set pack -> clear after terminal;
            # st_place (prompt mode): build_demo_pack clears (skipped), then
            # the run_mission clear — no pack is ever pushed for it.
            kinds = [c[0] for c in transport.calls]
            self.assertEqual(kinds, ["set", "clear", "clear", "clear"])
            set_k = [c[1] for c in transport.calls if c[0] == "set"]
            self.assertEqual(set_k, [2])

    def test_run_mission_auto_advances_immediately(self):
        package = _package()
        with tempfile.TemporaryDirectory() as tmp:
            for st in package["subtasks"]:
                for d in st.get("conditioning", {}).get("demonstrations", []):
                    _write_keyframe_png(Path(tmp) / f"k{d['episode_id']}.png", seed=1)
                    d["keyframes"] = [f"file://{tmp}/k{d['episode_id']}.png"]
            state = BridgeState(parse_dispatch_package(package))
            transport = _SequenceTransport()
            completed = run_mission(
                state, transport, k=4, frames_per_demo=6, k_max=4, advance_mode="auto",
            )
        self.assertEqual(completed, ["st_open", "st_place"])


if __name__ == "__main__":
    unittest.main()
