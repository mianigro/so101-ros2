"""Source-level contracts for simulation-only collision pad authoring."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT_XACRO = REPOSITORY_ROOT / "so101_description/urdf/so101_arm.urdf.xacro"
FOLLOWER_XACRO = (
    REPOSITORY_ROOT
    / "so101_description/urdf/end_effectors/so101_ee_follower.xacro"
)
ISAAC_IMPORTER = REPOSITORY_ROOT / "scripts/isaac_sim_teleop.py"
XACRO_NS = {"xacro": "http://www.ros.org/wiki/xacro"}


class ContactPadDescriptionTests(unittest.TestCase):
    def test_default_robot_description_keeps_simulation_pads_disabled(self):
        root = ET.parse(ROOT_XACRO).getroot()
        option = root.find("xacro:arg[@name='simulation_contact_pads']", XACRO_NS)
        self.assertIsNotNone(option)
        self.assertEqual(option.attrib["default"], "false")

    def test_pad_links_are_fixed_collision_only_analytic_boxes(self):
        root = ET.parse(FOLLOWER_XACRO).getroot()
        expected = {
            "fixed_jaw_contact_pad_link": (
                "0.0005 0.018 0.025",
                "-0.00805 -0.000218 -0.0925",
                "7.908333333e-09",
                "5.210416667e-09",
                "2.702083333e-09",
            ),
            "moving_jaw_contact_pad_link": (
                "0.0005 0.025 0.018",
                # Origin X calibrated so the pad face is flush with the
                # moving-jaw finger mesh at the grasp angle (see METHODOLOGY).
                "-0.00996 -0.0691 0.0190",
                "7.908333333e-09",
                "2.702083333e-09",
                "5.210416667e-09",
            ),
        }
        for link_name, (size, origin, ixx, iyy, izz) in expected.items():
            link = root.find(f".//link[@name='{link_name}']")
            self.assertIsNotNone(link)
            self.assertIsNone(link.find("visual"))
            self.assertEqual(link.find("./collision/geometry/box").attrib["size"], size)
            inertial = link.find("xacro:inertial_block", XACRO_NS)
            self.assertEqual(inertial.attrib["mass"], "0.0001")
            self.assertEqual(inertial.attrib["ixx"], ixx)
            self.assertEqual(inertial.attrib["iyy"], iyy)
            self.assertEqual(inertial.attrib["izz"], izz)

            joint_name = link_name.replace("_link", "_joint")
            joint = root.find(f".//joint[@name='{joint_name}']")
            self.assertEqual(joint.attrib["type"], "fixed")
            self.assertEqual(joint.find("origin").attrib["xyz"], origin)

    def test_isaac_expansion_enables_pads_without_merging_fixed_joints(self):
        source = ISAAC_IMPORTER.read_text(encoding="utf-8")
        self.assertIn('"simulation_contact_pads:=true"', source)
        self.assertIn("merge_fixed_joints=False", source)


if __name__ == "__main__":
    unittest.main()
