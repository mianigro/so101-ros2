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

"""M5 packaging tests: the so101_icl_bridge ROS package is well-formed.

No ROS install required — these check the files an ament_python build and
``ros2 launch`` need, plus the so101_icl path resolution the entry point
relies on. The live-robot bring-up itself remains a manual session.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PKG_DIR = Path(__file__).resolve().parents[1] / "so101_icl_bridge"


class TestPackageFiles(unittest.TestCase):
    def test_ament_python_layout(self):
        for rel in (
            "package.xml", "setup.py", "setup.cfg",
            "resource/so101_icl_bridge",
            "so101_icl_bridge/__init__.py",
            "so101_icl_bridge/bridge.py",
            "launch/bridge_icl.launch.py",
        ):
            self.assertTrue((PKG_DIR / rel).is_file(), f"missing {rel}")

    def test_package_xml_name_and_build_type(self):
        text = (PKG_DIR / "package.xml").read_text()
        self.assertIn("<name>so101_icl_bridge</name>", text)
        self.assertIn("<build_type>ament_python</build_type>", text)
        self.assertIn("rclpy", text)

    def test_entry_point_target_exists(self):
        setup = (PKG_DIR / "setup.py").read_text()
        self.assertIn("bridge_icl_node = so101_icl_bridge.bridge:main", setup)
        self.assertTrue((PKG_DIR / "so101_icl_bridge" / "bridge.py").is_file())

    def test_launch_file_declares_node_params(self):
        launch = (PKG_DIR / "launch" / "bridge_icl.launch.py").read_text()
        for param in ("api_base", "workspace", "mission_id", "demo_port",
                      "stats_path", "terminal_msg_type"):
            self.assertIn(param, launch)
        self.assertIn('package="so101_icl_bridge"', launch)


class TestPathResolution(unittest.TestCase):
    def test_sibling_resolution(self):
        # importing the package module without installing it
        spec = importlib.util.spec_from_file_location(
            "so101_icl_bridge._bridge_test", PKG_DIR / "so101_icl_bridge" / "bridge.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        resolved = module._so101_icl_path()
        # so101_icl is importable in the test env (sys.path above) -> no-op,
        # or the sibling repo dir -> must contain the bridge node module
        if resolved:
            self.assertTrue(
                (Path(resolved) / "so101_icl" / "bridge_icl_node.py").is_file(),
                f"resolved path {resolved} does not contain the node",
            )


if __name__ == "__main__":
    unittest.main()
