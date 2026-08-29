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

"""rclpy entry point for the ICL bridge node (ICL §8 M5).

The node core (selection, ordering, state machine, terminal monitor, demo
transport) lives in ``so101_icl.bridge_icl_node.BridgeICLNode`` and is
unit-tested; this package only adds the missing ROS wiring —
``rclpy.init``/spin/shutdown and an installable launch target — so the node
can run as ``ros2 run so101_icl_bridge bridge_icl_node`` /
``ros2 launch so101_icl_bridge bridge_icl.launch.py``.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    import rclpy

    sys.path.insert(0, _so101_icl_path())

    from so101_icl.bridge_icl_node import BridgeICLNode

    rclpy.init(args=argv)
    node = BridgeICLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.transport.clear()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _so101_icl_path() -> str:
    """Locate the ``so101_icl`` package directory.

    Resolution order: already importable (no-op), ``SO101_ICL_ROOT`` env
    var, then the sibling directory next to this package in the workspace
    (colcon build keeps the sources under ``src/``-style trees where the
    pixi checkout places ``so101_icl/so101_icl``).
    """
    import importlib.util
    import os
    from pathlib import Path

    if importlib.util.find_spec("so101_icl") is not None:
        return ""
    env = os.environ.get("SO101_ICL_ROOT")
    if env:
        return str(Path(env).expanduser())
    candidates = [
        # <repo>/so101_icl (dir containing the so101_icl python package)
        Path(__file__).resolve().parents[2] / "so101_icl",
    ]
    for path in candidates:
        if (path / "so101_icl" / "bridge_icl_node.py").is_file():
            return str(path)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
