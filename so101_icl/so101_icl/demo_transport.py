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

"""ZMQ side channel for demo packs (ICL §4.6).

The async inference client and the 30 Hz contract are untouched; the
**bridge node** drives this side channel once per subtask:

    bridge_icl_node ── set_demo_pack / clear / status ──► REP socket (8661)
                                                          │ calls the loaded
                                                          │ policy's cache API
    cameras/state @30Hz ──► async_inference_node ──ZMQ──► serve_icl (main port)

Wire format: a msgpack header frame followed by raw little-endian float32
array frames (same no-pickle convention as the main transport). Frames
arrive ALREADY preprocessed ([-1, 1], 224) and trajectories already
normalized with the ACTIVE stage's stats — the single shared image/normalization
path lives in :mod:`so101_icl.data` (``preprocess_demo_frames``,
``TrajNormalizer``) and is used by both training and the bridge.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8661

# Module-level holder fed by the engine hook; the transport is started before
# any policy exists (policies load on the first client handshake).
_current_policy: dict = {"policy": None}


def install_into_engine() -> None:
    """Track the InferenceEngine's loaded policy (called from serve_icl).

    Wraps ``load_policy``/``unload_policy`` on the class — policy_server
    itself is not modified.
    """
    from policy_server.inference_engine import InferenceEngine

    if getattr(InferenceEngine, "_icl_patched", False):
        return

    orig_load = InferenceEngine.load_policy
    orig_unload = InferenceEngine.unload_policy

    def load_policy(self, config):
        policy = orig_load(self, config)
        _current_policy["policy"] = policy
        return policy

    def unload_policy(self):
        _current_policy["policy"] = None
        return orig_unload(self)

    InferenceEngine.load_policy = load_policy
    InferenceEngine.unload_policy = unload_policy
    InferenceEngine._icl_patched = True
    logger.info("demo transport hooked into InferenceEngine")


class DemoTransportServer:
    """REP socket executing set/clear/status against the loaded policy."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.endpoint = f"tcp://{host}:{port}"
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(self.endpoint)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._last_encode_s = None

    def start(self) -> "DemoTransportServer":
        self._thread.start()
        logger.info("demo transport listening on %s", self.endpoint)
        return self

    def _serve(self) -> None:
        while True:
            try:
                frames = self._sock.recv_multipart()
                reply = self._handle(frames)
            except Exception as e:  # noqa: BLE001 — REP must always answer
                logger.exception("demo transport error")
                reply = {"status": "error", "error": str(e)}
            self._sock.send(msgpack.packb(reply, use_bin_type=True))

    def _handle(self, frames: list[bytes]) -> dict:
        header = msgpack.unpackb(frames[0], raw=False)
        policy = _current_policy["policy"]
        msg_type = header.get("type")

        if msg_type == "status":
            model = policy.model if policy is not None else None
            return {
                "status": "ok",
                "policy_loaded": policy is not None,
                "pack_set": bool(model and model.demo_pack_is_set),
                "last_encode_s": self._last_encode_s,
            }

        if policy is None:
            return {"status": "error", "error": "no policy loaded yet"}

        if msg_type == "clear":
            policy.model.clear_demo_pack()
            return {"status": "ok"}

        if msg_type == "set_demo_pack":
            start = time.perf_counter()
            k_max = int(header["k_max"])
            frame_shape = tuple(header["frame_shape"])       # e.g. [3, 224, 224]
            frames_per_demo = int(header["frames_per_demo"])
            expected_elems = k_max * frames_per_demo * int(np.prod(frame_shape))
            # 2 parts: frames only; 3 parts: frames + keypoint features (rev 5)
            if len(frames) not in (2, 3) or len(frames[1]) != expected_elems * 4:
                return {
                    "status": "error",
                    "error": (
                        f"expected one binary frame of {expected_elems} float32 "
                        f"values, got {len(frames) - 1} frame(s)"
                    ),
                }
            demo_frames = np.frombuffer(frames[1], dtype="<f4").reshape(
                k_max, frames_per_demo, *frame_shape
            )

            traj = np.zeros(
                (k_max, int(header.get("traj_steps", 16)), int(header.get("traj_dim", 64))),
                dtype=np.float32,
            )
            traj_ok = np.zeros(k_max, dtype=np.float32)
            if header.get("traj") is not None:
                traj[:] = np.asarray(header["traj"], dtype=np.float32).reshape(traj.shape)
            if header.get("traj_ok") is not None:
                traj_ok[:] = np.asarray(header["traj_ok"], dtype=np.float32)

            import torch

            mask = torch.zeros(k_max, dtype=torch.bool)
            mask[: int(header["k"])] = True
            kp = kp_ok = None
            if header.get("kp") is not None:
                # keypoint features ride as a second binary frame (rev 5)
                n_kp, kp_dim = int(header["n_kp"]), int(header["kp_dim"])
                expected_kp = k_max * frames_per_demo * n_kp * kp_dim
                if len(frames) != 3 or len(frames[2]) != expected_kp * 4:
                    return {
                        "status": "error",
                        "error": (
                            f"expected a binary keypoint frame of {expected_kp} "
                            f"float32 values, got {max(0, len(frames) - 2)} extra frame(s)"
                        ),
                    }
                kp = torch.from_numpy(
                    np.frombuffer(frames[2], dtype="<f4").copy().reshape(
                        k_max, frames_per_demo, n_kp, kp_dim
                    )
                )
                kp_ok = torch.zeros(k_max)
                ok_vals = np.asarray(header["kp_ok"], dtype=np.float32)
                kp_ok[: int(header["k"])] = torch.from_numpy(ok_vals[: int(header["k"])])
            set_kwargs = {}
            if kp is not None:
                set_kwargs = {"kp": kp, "kp_ok": kp_ok}
            policy.model.set_demo_pack(
                torch.from_numpy(demo_frames),
                mask,
                torch.from_numpy(traj),
                torch.from_numpy(traj_ok),
                **set_kwargs,
            )
            self._last_encode_s = time.perf_counter() - start
            return {"status": "ok", "encode_s": self._last_encode_s, "k": int(header["k"])}

        return {"status": "error", "error": f"unknown type {msg_type!r}"}


def maybe_spawn_demo_transport(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> DemoTransportServer | None:
    """Install the engine hook and start the side channel (serve_icl.py)."""
    try:
        install_into_engine()
    except ImportError:
        logger.warning("policy_server not importable; demo transport disabled")
        return None
    return DemoTransportServer(host, port).start()


class DemoTransportClient:
    """Client used by bridge_icl_node (and the tests)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.endpoint = f"tcp://{host}:{port}"
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, 10_000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def _call(self, header, arrays: list[np.ndarray] | None = None) -> dict:
        parts = [msgpack.packb(header, use_bin_type=True)]
        for arr in arrays or []:
            parts.append(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        self._sock.send_multipart(parts)
        return msgpack.unpackb(self._sock.recv(), raw=False)

    def set_demo_pack(
        self,
        demo_frames: np.ndarray,       # [k, F, 3, H, W] float32, already preprocessed
        traj: np.ndarray | None = None,  # [k, S, d] float32, already normalized
        traj_ok: np.ndarray | None = None,
        k_max: int = 4,
        *,
        kp: np.ndarray | None = None,    # [k, F, K, kp_dim] SIFT features (rev 5)
        kp_ok: np.ndarray | None = None, # [k] float 0/1
    ) -> dict:
        k, f = demo_frames.shape[:2]
        padded = np.zeros((k_max, f, *demo_frames.shape[2:]), dtype=np.float32)
        padded[:k] = demo_frames
        arrays = [padded.reshape(-1)]
        header_extra = {}
        if kp is not None:
            kp_padded = np.zeros((k_max, *kp.shape[1:]), dtype=np.float32)
            kp_padded[:k] = kp
            arrays.append(kp_padded.reshape(-1))
            if kp_ok is None:
                kp_ok = np.ones(k, dtype=np.float32)
            kp_ok_padded = np.zeros(k_max, dtype=np.float32)
            kp_ok_padded[:k] = kp_ok
            header_extra = {
                "kp": True,                       # signals the binary part exists
                "n_kp": int(kp.shape[2]),
                "kp_dim": int(kp.shape[3]),
                "kp_ok": kp_ok_padded.tolist(),
            }
        if traj is None:
            traj = np.zeros((k_max, 16, 64), dtype=np.float32)
        else:
            t = np.zeros((k_max, *traj.shape[1:]), dtype=np.float32)
            t[:k] = traj
            traj = t
        if traj_ok is None:
            traj_ok = np.zeros(k_max, dtype=np.float32)
        else:
            ok = np.zeros(k_max, dtype=np.float32)
            ok[:k] = traj_ok
            traj_ok = ok
        return self._call(
            {
                "type": "set_demo_pack",
                "k": int(k),
                "k_max": int(k_max),
                "frames_per_demo": int(f),
                "frame_shape": list(demo_frames.shape[2:]),
                "traj_steps": int(traj.shape[1]),
                "traj_dim": int(traj.shape[2]),
                "traj": traj.tolist(),
                "traj_ok": traj_ok.tolist(),
                **header_extra,
            },
            arrays=arrays,
        )

    def clear(self) -> dict:
        return self._call({"type": "clear"})

    def status(self) -> dict:
        return self._call({"type": "status"})


# ---------------------------------------------------------------------- #
# Stage stats transport (bridge-side trajectory normalization)            #
# ---------------------------------------------------------------------- #


def save_stage_stats(normalizer, path) -> None:
    """Serialize a TrajNormalizer's stats next to an adapter/serving ckpt."""
    stats = normalizer._step.stats  # noqa: SLF001 (pinned lerobot 0.6.1)
    out = {
        key: {stat: np.asarray(tensor).tolist() for stat, tensor in per_key.items()}
        for key, per_key in stats.items()
        if key in ("observation.state", "action")
    }
    from pathlib import Path

    Path(path).write_text(json.dumps(out))


def load_stage_stats(path) -> dict:
    import torch

    raw = json.loads(open(path).read())
    return {
        key: {stat: torch.tensor(vals, dtype=torch.float32) for stat, vals in per.items()}
        for key, per in raw.items()
    }
