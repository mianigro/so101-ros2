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

"""Chunk-latency measurement for the M4 gate (ICL §8: p95 within +20%).

Three pieces:

- :class:`LatencyRecorder` — timestamp bookkeeping + percentile summary;
- :class:`TimedRolloutClient` — wraps the sim rollout client and times
  every ``infer`` round trip (M3 campaign side effect);
- :func:`probe_policy_server` — a standalone probe against the running
  ``serve_icl`` policy server (ZMQ) using the robot stack's own transport,
  for the real-robot campaign where inference runs inside
  ``async_inference_node`` and cannot be instrumented from outside.
"""

from __future__ import annotations

import time


class LatencyRecorder:
    """Accumulates durations (seconds) and reports percentiles."""

    def __init__(self, name: str = ""):
        self.name = name
        self._samples: list[float] = []

    def record(self, seconds: float) -> None:
        self._samples.append(float(seconds))

    def timed(self) -> "_Timed":
        return _Timed(self)

    @property
    def samples(self) -> list[float]:
        return list(self._samples)

    def stats(self) -> dict:
        if not self._samples:
            return {"name": self.name, "n": 0}
        s = sorted(self._samples)

        def pct(p: float) -> float:  # linear interpolation (numpy-style)
            if len(s) == 1:
                return s[0]
            import math

            k = (len(s) - 1) * p / 100.0
            lo = math.floor(k)
            hi = min(lo + 1, len(s) - 1)
            return s[lo] + (s[hi] - s[lo]) * (k - lo)

        return {
            "name": self.name,
            "n": len(s),
            "mean_s": sum(s) / len(s),
            "p50_s": pct(50),
            "p95_s": pct(95),
            "max_s": s[-1],
        }

    def summary_line(self) -> str:
        st = self.stats()
        if st["n"] == 0:
            return f"{self.name or 'latency'}: no samples"
        return (f"{self.name or 'latency'}: n={st['n']} mean={st['mean_s']:.3f}s "
                f"p50={st['p50_s']:.3f}s p95={st['p95_s']:.3f}s max={st['max_s']:.3f}s")


class _Timed:
    """Context manager recording wall time into a recorder."""

    def __init__(self, recorder: LatencyRecorder):
        self._recorder = recorder
        self._start = 0.0

    def __enter__(self) -> "_Timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self._recorder.record(time.perf_counter() - self._start)


class TimedRolloutClient:
    """Transparent wrapper timing every ``infer`` of a rollout client."""

    def __init__(self, client, recorder: LatencyRecorder | None = None):
        self._client = client
        self.recorder = recorder or LatencyRecorder("chunk")

    @property
    def chunk_size(self):
        return self._client.chunk_size

    @property
    def action_dim(self):
        return self._client.action_dim

    def hello(self, *args, **kwargs):
        return self._client.hello(*args, **kwargs)

    def infer(self, *args, **kwargs):
        with self.recorder.timed():
            return self._client.infer(*args, **kwargs)

    def close(self):
        return self._client.close()

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._client.__exit__(*exc_info)


def latency_gate(recorders: dict[str, LatencyRecorder], tolerance: float = 0.20) -> str:
    """M4 gate line: full_icl p95 within +tolerance of bare_prompt p95."""
    lines = []
    base = recorders.get("bare_prompt") or recorders.get("prompt_enriched")
    full = recorders.get("full_icl")
    if base is None or full is None or not base.samples or not full.samples:
        return "latency gate: not evaluable (missing conditions or samples)"
    base_p95 = base.stats()["p95_s"]
    full_p95 = full.stats()["p95_s"]
    ratio = full_p95 / base_p95 if base_p95 > 0 else float("inf")
    ok = ratio <= 1.0 + tolerance
    lines.append(
        f"latency gate: full_icl p95 {full_p95:.3f}s vs baseline p95 {base_p95:.3f}s "
        f"(ratio {ratio:.2f}, gate <= {1.0 + tolerance:.2f}): "
        + ("PASS" if ok else "FAIL")
    )
    return "\n".join(lines)


def probe_policy_server(
    server_address: str,
    *,
    repo_id: str,
    n: int = 30,
    actions_per_chunk: int = 16,
    task: str = "pick up the cube and place it in the cup",
    seed: int = 0,
) -> LatencyRecorder:
    """Time ``n`` synthetic infer round trips against the policy server.

    Uses the robot stack's own ZMQ transport (``so101_inference``) so the
    measurement covers exactly what ``async_inference_node`` pays per
    chunk: serialization, socket, preprocess, forward, postprocess.
    Synthetic observations come from a fixed seed — absolute values are
    comparable across conditions, which is what the gate needs.
    """
    import numpy as np

    from so101_inference.transport.zmq_transport import ZmqTransport

    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": None}}
    for cam in ("wrist", "overhead_1", "overhead_2"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    rng = np.random.default_rng(seed)
    obs = {
        f"observation.images.{cam}": (rng.random((3, 480, 640), dtype=np.float32) * 255)
        .astype(np.uint8)
        for cam in ("wrist", "overhead_1", "overhead_2")
    }
    obs["observation.state"] = rng.random((6,), dtype=np.float32) * 2 - 1
    obs["task"] = task

    from lerobot.async_inference.helpers import RemotePolicyConfig

    recorder = LatencyRecorder(f"probe:{server_address}")
    transport = ZmqTransport(server_address=server_address)
    if not transport.connect() or not transport.handshake():
        raise RuntimeError(f"cannot connect to policy server at {server_address}")
    transport.send_policy_config(RemotePolicyConfig(
        policy_type="pi05_icl",
        pretrained_name_or_path=repo_id,
        lerobot_features=features,
        actions_per_chunk=actions_per_chunk,
        device="cuda",
    ))
    try:
        for i in range(n):
            from lerobot.async_inference.helpers import TimedObservation

            timed_obs = TimedObservation(
                {k: (v if not hasattr(v, "unsqueeze") else v) for k, v in obs.items()},
                timestep=i, timestamp=float(i) / 30.0, must_go=True,
            )
            with recorder.timed():
                transport.infer(timed_obs)
    finally:
        transport.close()
    return recorder
