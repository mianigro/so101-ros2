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

"""End-to-end bridge integration test against a STUB dispatch service.

Exercises the full never-run M5 path (ICL §11.3) without ROS or a GPU:

    stub HTTP dispatch ──► bridge main() ──► real ZMQ demo side channel
                            (poll, order,        (DemoTransportServer +
                             select, fetch)       DemoTransportClient, real
                                                 msgpack+float32 wire)
                                                 ──► fake policy model

Uses ``file://`` keyframes so no network beyond loopback is involved.
"""

import json
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from so101_icl import demo_transport  # noqa: E402

try:
    import zmq  # noqa: F401

    ZMQ_OK = True
except ImportError:
    ZMQ_OK = False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeModel:
    """Duck-typed PI05ICLCore: records pack lifecycle calls."""

    def __init__(self):
        self.calls = []
        self.demo_pack_is_set = False
        self.retrieval_is_set = False

    def set_demo_pack(self, frames, mask, traj, traj_ok):
        self.calls.append(("set_demo_pack", tuple(frames.shape), mask.sum().item()))
        self.demo_pack_is_set = True

    def clear_demo_pack(self):
        self.calls.append(("clear_demo_pack",))
        self.demo_pack_is_set = False

    def set_retrieval_index(self, index, lam=5.0):
        self.calls.append(("set_retrieval_index",))
        self.retrieval_is_set = True

    def clear_retrieval_index(self):
        self.calls.append(("clear_retrieval_index",))
        self.retrieval_is_set = False


class _FakePolicy:
    def __init__(self):
        self.model = _FakeModel()


def _write_keyframe_png(path: Path, seed: int):
    img = (np.random.default_rng(seed).random((480, 640, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _package(tmp: str) -> dict:
    return {
        "mission_id": "m-itest",
        "subtasks": [
            {
                "id": "st_pick",
                "depends_on": [],
                "action_spec": {"prompt": "pick the red cube"},
                "conditioning": {"mode": "demo", "demonstrations": [
                    {"episode_id": 1, "outcome": "success", "prov_id": 10,
                     "keyframes": [f"file://{tmp}/kf1.png", f"file://{tmp}/kf2.png"]},
                    {"episode_id": 2, "outcome": "success", "prov_id": 20,
                     "keyframes": [f"file://{tmp}/kf3.png"]},
                ]},
            },
            {
                "id": "st_place",
                "depends_on": ["st_pick"],
                "action_spec": {"prompt": "place it in the bin"},
                "conditioning": {"mode": "prompt", "demonstrations": []},
            },
        ],
    }


@unittest.skipUnless(ZMQ_OK, "zmq not available")
class TestBridgeIntegration(unittest.TestCase):
    """Dispatch HTTP -> bridge main() -> real side channel -> fake model."""

    def test_full_mission_through_stub_dispatch(self):
        from so101_icl.bridge_icl_node import main as bridge_main
        from so101_icl.demo_transport import (
            DemoTransportClient,
            DemoTransportServer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            for i in (1, 2, 3):
                _write_keyframe_png(Path(tmp) / f"kf{i}.png", seed=i)
            package = _package(tmp)

            # --- stub dispatch service (the Bridge Robot API contract) ---
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    if self.path.endswith("/dispatch-package"):
                        body = json.dumps(package).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_error(404)

                def log_message(self, *args):  # silence the test log
                    pass

            httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            http_port = httpd.server_address[1]
            threading.Thread(target=httpd.serve_forever, daemon=True).start()

            # --- real demo side channel against a fake policy ---
            fake = _FakePolicy()
            demo_transport._current_policy["policy"] = fake
            demo_port = _free_port()
            server = DemoTransportServer("127.0.0.1", demo_port).start()

            try:
                rc = bridge_main([
                    "--api-base", f"http://127.0.0.1:{http_port}",
                    "--workspace", "ws", "--mission-id", "m-itest",
                    "--demo-port", str(demo_port),
                    "--k", "2", "--k-max", "4", "--frames-per-demo", "6",
                    "--advance-mode", "auto", "--once",
                ])
                self.assertEqual(rc, 0)

                kinds = [c[0] for c in fake.model.calls]
                # st_pick: pack of 2 demos pushed, then cleared on advance;
                # st_place (prompt mode) is skipped -> clear only, plus the
                # final clear-on-advance.
                self.assertEqual(kinds, [
                    "set_demo_pack", "clear_demo_pack",
                    "clear_demo_pack", "clear_demo_pack",
                ])
                set_call = fake.model.calls[0]
                self.assertEqual(set_call[1], (4, 6, 3, 224, 224))  # k_max slots
                self.assertEqual(set_call[2], 2)                     # k demos

                # the side channel itself reports the cleared state
                client = DemoTransportClient("127.0.0.1", demo_port)
                status = client.status()
                self.assertTrue(status["policy_loaded"])
                self.assertFalse(status["pack_set"])
            finally:
                demo_transport._current_policy["policy"] = None
                httpd.shutdown()
                server._thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
