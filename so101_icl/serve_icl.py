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

"""Serve pi05_icl through the standard policy_server ZMQ main.

1. registers ``"pi05_icl"`` in the lerobot policy registry (no lerobot edits),
2. starts the demo-pack side channel (ICL_DEMO_PORT, default 8661),
3. runs ``policy_server.zmq_server.main()`` — the client handshake supplies
   ``policy_type: "pi05_icl"`` at runtime, exactly like any other policy.

Pixi task ``icl_server`` wraps this. The async inference node, the 30 Hz
contract and the chunking engine are untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))  # policy_server is a flat-layout package dir

import so101_icl.registration  # noqa: F401,E402  (registers "pi05_icl")

from so101_icl.demo_transport import maybe_spawn_demo_transport  # noqa: E402


def main() -> int:
    demo_port = int(os.environ.get("ICL_DEMO_PORT", "8661"))
    if os.environ.get("ICL_DEMO_TRANSPORT", "1") == "1":
        maybe_spawn_demo_transport(port=demo_port)
    from policy_server import zmq_server

    return zmq_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
