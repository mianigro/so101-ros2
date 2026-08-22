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

"""Length-prefixed pickle protocol between the sim rollout client and the
rollout policy server.

The client side runs inside Isaac Sim's own Python, where lerobot is not
installed — hence stdlib + numpy only, and hence a purpose-built protocol
instead of reusing ``policy_server``'s lerobot-typed wire format.  The socket
is expected to live on a trusted host (localhost / private LAN): pickled
objects are only as safe as the peer.

Messages are dicts with a ``type`` key:

* ``hello``  {features, actions_per_chunk}  -> ``welcome`` {chunk, action_dim}
  (or ``error`` {message})
* ``infer``  {images: {key: uint8 [B,H,W,3]}, state: [B,6], task: str}
  -> ``chunk`` {actions: float32 [B, chunk, action_dim]}  (or ``error``)
* ``bye``    -> connection closed
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Callable, Dict, Optional

import numpy as np

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8660
_LENGTH_HEADER = struct.Struct(">Q")
_MAX_MESSAGE_BYTES = 1 << 30  # 1 GiB safety cap (a 32-env batch is ~180 MiB)


class WireError(RuntimeError):
    """Raised on protocol, timeout, or peer-error failures."""


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LENGTH_HEADER.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    header = _recv_exactly(sock, _LENGTH_HEADER.size)
    (length,) = _LENGTH_HEADER.unpack(header)
    if length > _MAX_MESSAGE_BYTES:
        raise WireError(f"peer message of {length} bytes exceeds cap")
    payload = _recv_exactly(sock, length)
    try:
        return pickle.loads(payload)
    except Exception as exc:  # pragma: no cover - corrupt peer data
        raise WireError(f"failed to unpickle peer message: {exc}") from exc


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise WireError("peer closed the connection mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _expect(message: Any, expected_type: str) -> Dict[str, Any]:
    if not isinstance(message, dict):
        raise WireError(f"expected dict message, got {type(message)!r}")
    msg_type = message.get("type")
    if msg_type == "error":
        raise WireError(f"server error: {message.get('message', '<no message>')}")
    if msg_type != expected_type:
        raise WireError(f"expected {expected_type!r} message, got {msg_type!r}")
    return message


class RolloutClient:
    """Blocking request/response client for the rollout policy server."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout_s: float = 120.0) -> None:
        self._address = (host, port)
        self._sock: Optional[socket.socket] = None
        self._timeout_s = timeout_s
        self.chunk_size: Optional[int] = None
        self.action_dim: Optional[int] = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            self._sock = socket.create_connection(self._address, timeout=10.0)
        except OSError as exc:
            raise WireError(
                f"cannot reach rollout policy server at {self._address}: {exc}"
            ) from exc
        self._sock.settimeout(self._timeout_s)

    def hello(self, features: Dict[str, dict], actions_per_chunk: int,
              task: str, image_key_map: Optional[Dict[str, str]] = None,
              action_dim: Optional[int] = None) -> Dict[str, Any]:
        self.connect()
        assert self._sock is not None
        send_msg(self._sock, {
            "type": "hello",
            "features": features,
            "actions_per_chunk": int(actions_per_chunk),
            "task": task,
            # Optional client->policy camera key remap (zero-shot base
            # checkpoints use openpi-style keys like base_0_rgb) and the
            # client's true action dimension (base pads to 32).
            "image_key_map": image_key_map or {},
            "action_dim": int(action_dim) if action_dim else None,
        })
        reply = _expect(recv_msg(self._sock), "welcome")
        self.chunk_size = int(reply["chunk"])
        self.action_dim = int(reply["action_dim"])
        return reply

    def infer(self, images: Dict[str, np.ndarray], state: np.ndarray,
              task: str) -> np.ndarray:
        """Request action chunks for a batch of environments.

        ``images`` maps dataset feature key -> uint8 [B, H, W, 3]; ``state``
        is [B, 6] float in dataset units.  Returns float32 [B, chunk, action_dim].
        """
        if self._sock is None:
            raise WireError("not connected; call hello() first")
        send_msg(self._sock, {
            "type": "infer",
            "images": {key: np.ascontiguousarray(value)
                       for key, value in images.items()},
            "state": np.ascontiguousarray(np.asarray(state, dtype=np.float64)),
            "task": task,
        })
        reply = _expect(recv_msg(self._sock), "chunk")
        actions = np.asarray(reply["actions"], dtype=np.float32)
        if actions.ndim != 3:
            raise WireError(f"expected [B, chunk, dim] actions, got {actions.shape}")
        return actions

    def close(self) -> None:
        if self._sock is not None:
            try:
                send_msg(self._sock, {"type": "bye"})
            except OSError:
                pass
            finally:
                self._sock.close()
                self._sock = None

    def __enter__(self) -> "RolloutClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def serve_forever(host: str, port: int, handler: Callable[[socket.socket], None],
                  backlog: int = 1) -> None:
    """Accept-loop for the server side; one client connection at a time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(backlog)
        print(f"rollout policy server listening on {host}:{port}", flush=True)
        while True:
            conn, addr = server.accept()
            print(f"client connected: {addr}", flush=True)
            try:
                handler(conn)
            except WireError as exc:
                print(f"client {addr} dropped: {exc}", flush=True)
            finally:
                conn.close()
                print(f"client disconnected: {addr}", flush=True)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "RolloutClient",
    "WireError",
    "recv_msg",
    "send_msg",
    "serve_forever",
]
