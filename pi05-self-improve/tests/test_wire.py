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

"""Wire protocol round-trip tests (client + server in-process, real sockets)."""

from __future__ import annotations

import threading
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05_selfimprove import wire


class _EchoHandler:
    """Minimal stand-in for the policy server used for protocol tests."""

    def __init__(self) -> None:
        self.seen: list = []

    def __call__(self, sock) -> None:
        while True:
            message = wire.recv_msg(sock)
            if message.get("type") == "bye":
                return
            self.seen.append(message)
            if message["type"] == "hello":
                wire.send_msg(sock, {"type": "welcome", "chunk": 4, "action_dim": 6})
            elif message["type"] == "infer":
                batch = message["state"].shape[0]
                wire.send_msg(sock, {
                    "type": "chunk",
                    "actions": np.zeros((batch, 4, 6), dtype=np.float32) + 0.5,
                })


class WireTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        handler = _EchoHandler()
        bound = threading.Event()
        port_holder: list = []

        def accept_loop() -> None:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("127.0.0.1", 0))
                server.listen(1)
                port_holder.append(server.getsockname()[1])
                bound.set()
                conn, _ = server.accept()
                try:
                    handler(conn)
                finally:
                    conn.close()

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        bound.wait(timeout=5.0)
        port = port_holder[0]

        images = {
            "observation.images.wrist": np.zeros((2, 12, 16, 3), dtype=np.uint8),
            "observation.images.overhead_1": np.full(
                (2, 12, 16, 3), 255, dtype=np.uint8),
        }
        state = np.array([[0.1] * 6, [-0.2] * 6])

        with wire.RolloutClient("127.0.0.1", port, timeout_s=5.0) as client:
            welcome = client.hello({"observation.state": {"shape": (6,)}}, 4,
                                   "do the task")
            self.assertEqual(welcome["chunk"], 4)
            self.assertEqual(client.chunk_size, 4)
            chunk = client.infer(images, state, "do the task")
            self.assertEqual(chunk.shape, (2, 4, 6))
            self.assertTrue(np.allclose(chunk, 0.5))

        self.assertEqual(len(handler.seen), 2)
        hello, infer = handler.seen
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(infer["type"], "infer")
        self.assertEqual(
            infer["images"]["observation.images.wrist"].dtype, np.uint8)
        np.testing.assert_array_equal(
            infer["images"]["observation.images.wrist"], images["observation.images.wrist"])
        np.testing.assert_array_equal(infer["state"], state)

    def test_error_message_raises(self) -> None:
        import socket

        def error_handler(sock) -> None:
            message = wire.recv_msg(sock)
            wire.send_msg(sock, {"type": "error",
                                 "message": "policy not loaded"})

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            accept = threading.Thread(
                target=lambda: error_handler(server.accept()[0]), daemon=True)
            accept.start()
            client = wire.RolloutClient("127.0.0.1", port, timeout_s=5.0)
            with self.assertRaises(wire.WireError) as ctx:
                client.hello({}, 4, "task")
            self.assertIn("policy not loaded", str(ctx.exception))
            client.close()


if __name__ == "__main__":
    unittest.main()
