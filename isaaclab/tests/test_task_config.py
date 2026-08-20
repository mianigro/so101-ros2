"""Configuration contract tests that do not launch Isaac Sim."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium as gym

from so101_rl.camera_profile import (
    POLICY_FREQUENCY_HZ,
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
    camera_profile_sha256,
    load_camera_profile,
    look_at_opengl_xyzw,
    wxyz_to_xyzw,
)
from so101_rl.tasks.common import SO101VisualObservationsCfg
from so101_rl.tasks.object_in_cup.object_in_cup_env_cfg import (
    SO101ObjectInCupVisionEnvCfg,
    SO101ObjectInCupVisionFixedEnvCfg,
)
from so101_rl.visual_contract import (
    SO101_ACTOR_OBSERVATION_GROUPS,
    SO101_JOINT_NAMES,
)


_MANIFEST = {
    "geometry": {
        "cube_extents_m": [0.025, 0.025, 0.025],
        "success_xy_tolerance_m": 0.0038,
        "success_center_z_min_m": 0.0145,
        "success_center_z_max_m": 0.0385,
    }
}


class TaskConfigTests(unittest.TestCase):
    def _config(self, fixed: bool = False):
        config_type = (
            SO101ObjectInCupVisionFixedEnvCfg
            if fixed
            else SO101ObjectInCupVisionEnvCfg
        )
        with patch(
            "so101_rl.tasks.object_in_cup.object_in_cup_env_cfg.load_asset_manifest",
            return_value=_MANIFEST,
        ):
            return config_type()

    def test_only_visual_tasks_are_registered(self):
        task_ids = {
            spec.id for spec in gym.registry.values() if spec.id.startswith("SO101-")
        }
        self.assertEqual(
            task_ids,
            {
                "SO101-Object-In-Cup-Vision-Fixed-v0",
                "SO101-Object-In-Cup-Vision-v0",
            },
        )

    def test_control_rate_action_contract_and_geometry_binding(self):
        cfg = self._config()
        cfg.validate()

        self.assertAlmostEqual(cfg.sim.dt, 1.0 / 120.0)
        self.assertEqual(cfg.decimation, 4)
        self.assertAlmostEqual(cfg.sim.dt * cfg.decimation, 1.0 / 30.0)
        self.assertEqual(tuple(cfg.actions.joint_delta.joint_names), SO101_JOINT_NAMES)
        self.assertTrue(cfg.actions.joint_delta.preserve_order)
        self.assertEqual(
            cfg.actions.joint_delta.clip["shoulder_.*|elbow_flex|wrist_.*"],
            (-1.0 / 30.0, 1.0 / 30.0),
        )
        self.assertEqual(cfg.actions.joint_delta.clip["gripper"], (-0.10, 0.10))
        self.assertEqual(cfg.terminations.success.params["required_steps"], 15)
        self.assertEqual(cfg.events.reset_layout.params["curriculum_steps"], 45_000_000)
        self.assertEqual(cfg.terminations.success.params["xy_tolerance"], 0.0038)
        self.assertEqual(
            tuple(cfg.rewards.__dataclass_fields__)[:6],
            (
                "approach",
                "lift_progress",
                "transport",
                "insertion",
                "release",
                "stable",
            ),
        )
        self.assertEqual(cfg.rewards.approach.weight, 1.0)
        self.assertEqual(cfg.rewards.approach.params["position_scale"], 0.15)
        # Retries after a miss re-earn the approach budget at a discount.
        self.assertEqual(cfg.rewards.approach.params["retry_radius"], 0.06)
        self.assertEqual(cfg.rewards.approach.params["retry_discount"], 0.5)
        self.assertEqual(cfg.rewards.lift_progress.weight, 1.0)
        self.assertEqual(
            cfg.rewards.lift_progress.params["object_rest_height"], 0.0125
        )
        self.assertEqual(cfg.rewards.lift_progress.params["height_scale"], 0.05)
        # Outcome-only rewards: no contact-gated params on any term.
        self.assertNotIn("force_threshold", cfg.rewards.lift_progress.params)
        self.assertNotIn("force_threshold", cfg.rewards.approach.params)
        self.assertAlmostEqual(
            cfg.rewards.transport.params["minimum_height"], 0.0135
        )
        # Transport credit is conditioned on lift height so pushing the cube
        # cannot farm it.
        self.assertAlmostEqual(cfg.rewards.transport.params["lift_height"], 0.03)
        # The pincer contact sensors are removed from this task's scene.
        self.assertFalse(hasattr(cfg.scene, "fixed_jaw_contact"))
        self.assertFalse(hasattr(cfg.scene, "moving_jaw_contact"))
        self.assertEqual(
            cfg.scene.ee_frame.target_frames[0].offset.pos,
            (0.0052, -0.000218, -0.0925),
        )

    def test_visual_observation_and_camera_contract(self):
        cfg = self._config()
        cfg.validate()

        common_observations = SO101VisualObservationsCfg()
        self.assertEqual(
            tuple(common_observations.__dataclass_fields__),
            SO101_ACTOR_OBSERVATION_GROUPS,
        )

        self.assertEqual(
            tuple(
                cfg.observations.joint_state.absolute_joint_positions.params[
                    "asset_cfg"
                ].joint_names
            ),
            SO101_JOINT_NAMES,
        )
        self.assertEqual(cfg.scene.wrist_camera.width, POLICY_IMAGE_WIDTH)
        self.assertEqual(cfg.scene.wrist_camera.height, POLICY_IMAGE_HEIGHT)
        self.assertEqual(POLICY_FREQUENCY_HZ, 30.0)
        self.assertAlmostEqual(cfg.scene.wrist_camera.update_period, 1.0 / 30.0)
        for name in ("wrist", "overhead_1", "overhead_2"):
            camera = getattr(cfg.scene, f"{name}_camera")
            camera_prim = getattr(cfg.scene, f"{name}_camera_prim")
            housing = getattr(cfg.scene, f"{name}_housing")
            self.assertIsNone(camera.spawn)
            self.assertEqual(camera_prim.prim_path, camera.prim_path)
            self.assertTrue(housing.prim_path.startswith(f"{camera.prim_path}/"))
            scene_fields = list(cfg.scene.__dict__)
            self.assertLess(
                scene_fields.index(f"{name}_camera_prim"),
                scene_fields.index(f"{name}_housing"),
            )
        self.assertFalse(cfg.scene.replicate_physics)
        self.assertEqual(cfg.actions.joint_delta.max_delay_steps, 1)

    def test_yaml_quaternion_and_overhead_optical_axis_contract(self):
        profile = load_camera_profile()
        wrist_wxyz = profile["cameras"]["wrist"]["orientation_wxyz"]
        self.assertEqual(
            wxyz_to_xyzw(wrist_wxyz),
            (wrist_wxyz[1], wrist_wxyz[2], wrist_wxyz[3], wrist_wxyz[0]),
        )
        overhead = profile["cameras"]["overhead_1"]
        quaternion = look_at_opengl_xyzw(
            overhead["position_m"], overhead["look_at_m"]
        )
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0, places=6)
        self.assertEqual(len(camera_profile_sha256()), 64)

    def test_fixed_visual_task_removes_randomization_and_latency(self):
        cfg = self._config(fixed=True)
        cfg.validate()
        self.assertEqual(cfg.actions.joint_delta.max_delay_steps, 0)
        self.assertIsNone(cfg.events.actuator_response)
        self.assertIsNone(cfg.events.randomize_cameras)
        self.assertIsNone(cfg.events.object_visual)
        self.assertIsNone(cfg.events.table_visual)
        self.assertIsNone(cfg.events.lighting)
        self.assertEqual(cfg.events.reset_robot.params["position_range"], (0.0, 0.0))
        self.assertIsNone(cfg.observations.joint_state.absolute_joint_positions.noise)
        self.assertEqual(
            cfg.events.reset_layout.params["object_xy_range_full"], (0.0, 0.0)
        )
        self.assertFalse(cfg.observations.wrist.rgb.params["randomize"])


if __name__ == "__main__":
    unittest.main()
