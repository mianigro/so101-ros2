#!/usr/bin/env python3
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

"""Rollout-wire policy server for ``pi05_icl`` (the M3 sim campaign server).

The sim rollout client speaks ``self_improve.wire`` (not policy_server's
ZMQ format), and ``self_improve.rollout_server`` only whitelists
pi05/pi0/smolvla. This wrapper:

1. registers ``pi05_icl`` in the lerobot registry (``so101_icl.registration``);
2. extends the whitelist without touching self-improve files;
3. publishes the loaded policy to the demo transport so the side channel's
   ``set_demo_pack``/``clear`` reach the live model (the sim-side twin of
   ``demo_transport.install_into_engine``);
4. spawns the demo transport on ``ICL_DEMO_PORT`` (default 8661), then
   delegates to the stock ``self_improve.rollout_server`` main.

Run inside the pixi ``lerobot`` env, serving a SAVING checkpoint from
``save_serving_checkpoint``::

    pixi run -e lerobot python so101_icl/serve_rollout_icl.py \\
        --repo-id so101_icl/runs/icl_finetune_so101_v1/serving \\
        --policy-type pi05_icl --host 0.0.0.0 --port 8660
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "self-improve"))

import so101_icl.registration  # noqa: F401,E402 — registers "pi05_icl"
from so101_icl import demo_transport  # noqa: E402

import self_improve.rollout_server as rollout_server  # noqa: E402

if "pi05_icl" not in rollout_server.SUPPORTED_POLICIES:
    rollout_server.SUPPORTED_POLICIES = (*rollout_server.SUPPORTED_POLICIES, "pi05_icl")

_orig_load = rollout_server.RolloutPolicyServer.load


def _publishing_load(self):
    _orig_load(self)
    # Same handoff the InferenceEngine patch performs for the ZMQ server.
    demo_transport._current_policy["policy"] = self.policy  # noqa: SLF001


rollout_server.RolloutPolicyServer.load = _publishing_load


def main() -> int:
    host = demo_transport.DEFAULT_HOST
    port = demo_transport.DEFAULT_PORT
    import os

    if os.environ.get("ICL_DEMO_TRANSPORT") != "0":
        host = os.environ.get("ICL_DEMO_HOST", host)
        port = int(os.environ.get("ICL_DEMO_PORT", port))
        demo_transport.maybe_spawn_demo_transport(host, port)
    if "--policy-type" not in sys.argv:
        # the stock main defaults to plain pi05; ICL serving never is
        sys.argv += ["--policy-type", "pi05_icl"]
    return rollout_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
