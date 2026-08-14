"""Focused tests for task-asset geometry and authored collision contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from so101_rl.asset_prep import (
    inspect_sources,
    prepare_assets,
    validate_generated_assets,
)
from so101_rl.paths import CUBE_STL_PATH, CUP_STL_PATH


class AssetPreparationTests(unittest.TestCase):
    def test_supplied_object_fits_the_open_cup(self):
        geometry = inspect_sources(CUBE_STL_PATH, CUP_STL_PATH)
        self.assertGreater(geometry.success_xy_tolerance_m, 0.003)
        self.assertLess(geometry.cube_horizontal_radius_m, geometry.cup_inner_radius_m)
        self.assertAlmostEqual(float(geometry.cube.extents[0]), 0.025, places=5)

    def test_authored_usd_uses_compound_convex_open_cup_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_dir = Path(temporary)
            robot_usd = temporary_dir / "robot.usda"
            robot_usd.write_text('#usda 1.0\ndef Xform "Robot" {}\n', encoding="utf-8")
            output_dir = temporary_dir / "prepared"
            manifest = prepare_assets(robot_usd=robot_usd, output_dir=output_dir)
            validated = validate_generated_assets(output_dir)

            self.assertEqual(validated, manifest)
            self.assertEqual(manifest["collision"]["cup"], "coacd_compound_convex")
            self.assertGreater(manifest["collision"]["cup_hull_count"], 1)
            self.assertLessEqual(
                manifest["collision"]["cup_hull_count"],
                manifest["collision"]["coacd_max_hulls"],
            )


if __name__ == "__main__":
    unittest.main()
