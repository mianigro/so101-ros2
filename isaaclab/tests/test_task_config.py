"""Configuration contract tests that do not launch Isaac Sim."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium as gym

from so101_rl.camera_profile import POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH
from so101_rl.tasks.object_in_cup.agents.rsl_rl_vision_ppo_cfg import (
    SO101ObjectInCupVisionPPOCfg,
)
from so101_rl.tasks.object_in_cup.mdp.critic_observations import CRITIC_STATE_DIM
from so101_rl.tasks.object_in_cup.object_in_cup_env_cfg import SO101_JOINTS
from so101_rl.tasks.object_in_cup.vision_env_cfg import (
    SO101ObjectInCupVisionEnvCfg,
    SO101ObjectInCupVisionFixedEnvCfg,
)


_MANIFEST = {
    "geometry": {
        "success_xy_tolerance_m": 0.0038,
        "success_center_z_min_m": 0.0145,
        "success_center_z_max_m": 0.0385,
    }
}


class TaskConfigTests(unittest.TestCase):
    def test_only_visual_tasks_are_registered(self):
        task_ids = {
            spec.id
            for spec in gym.registry.values()
            if spec.id.startswith("SO101-Object-In-Cup")
        }
        self.assertEqual(
            task_ids,
            {
                "SO101-Object-In-Cup-Vision-Fixed-v0",
                "SO101-Object-In-Cup-Vision-v0",
            },
        )

    def test_control_rate_action_contract_and_geometry_binding(self):
        with patch(
            "so101_rl.tasks.object_in_cup.object_in_cup_env_cfg.load_asset_manifest",
            return_value=_MANIFEST,
        ):
            cfg = SO101ObjectInCupVisionEnvCfg()
        cfg.validate()
        agent = SO101ObjectInCupVisionPPOCfg()

        self.assertEqual(cfg.sim.dt, 0.01)
        self.assertEqual(cfg.decimation, 5)
        self.assertEqual(cfg.sim.dt * cfg.decimation, 0.05)
        self.assertEqual(tuple(cfg.actions.joint_delta.joint_names), SO101_JOINTS)
        self.assertTrue(cfg.actions.joint_delta.preserve_order)
        self.assertEqual(
            cfg.actions.joint_delta.clip["shoulder_.*|elbow_flex|wrist_.*"],
            (-0.05, 0.05),
        )
        self.assertEqual(cfg.actions.joint_delta.clip["gripper"], (-0.15, 0.15))
        self.assertEqual(agent.clip_actions, 1.0)
        self.assertEqual(cfg.terminations.success.params["required_steps"], 10)
        self.assertEqual(cfg.terminations.success.params["xy_tolerance"], 0.0038)

    def test_visual_actor_and_simulator_critic_contract(self):
        with patch(
            "so101_rl.tasks.object_in_cup.object_in_cup_env_cfg.load_asset_manifest",
            return_value=_MANIFEST,
        ):
            cfg = SO101ObjectInCupVisionEnvCfg()
        cfg.validate()
        agent = SO101ObjectInCupVisionPPOCfg()

        self.assertEqual(
            agent.obs_groups["actor"],
            ["joint_state", "wrist", "overhead_1", "overhead_2"],
        )
        self.assertEqual(agent.obs_groups["critic"], ["critic_state"])
        self.assertFalse(cfg.observations.critic_state.enable_corruption)
        self.assertEqual(CRITIC_STATE_DIM, 34)
        self.assertEqual(
            tuple(cfg.observations.joint_state.absolute_joint_positions.params["asset_cfg"].joint_names),
            SO101_JOINTS,
        )
        self.assertEqual(cfg.scene.wrist_camera.width, POLICY_IMAGE_WIDTH)
        self.assertEqual(cfg.scene.wrist_camera.height, POLICY_IMAGE_HEIGHT)
        self.assertEqual(cfg.scene.wrist_camera.update_period, 0.05)
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
        self.assertEqual(agent.num_steps_per_env, 32)
        self.assertEqual(agent.algorithm.learning_rate, 7.0e-5)
        self.assertEqual(agent.algorithm.schedule, "fixed")
        self.assertEqual(agent.algorithm.num_mini_batches, 8)
        self.assertEqual(agent.actor.hidden_dims, [512, 256, 128])
        self.assertEqual(agent.critic.hidden_dims, [256, 256, 128])
        self.assertEqual(agent.critic.class_name, "MLPModel")
        self.assertIsNone(agent.critic.distribution_cfg)

    def test_fixed_visual_task_removes_randomization_and_latency(self):
        with patch(
            "so101_rl.tasks.object_in_cup.object_in_cup_env_cfg.load_asset_manifest",
            return_value=_MANIFEST,
        ):
            cfg = SO101ObjectInCupVisionFixedEnvCfg()
        cfg.validate()
        self.assertEqual(cfg.actions.joint_delta.max_delay_steps, 0)
        self.assertIsNone(cfg.events.randomize_cameras)
        self.assertIsNone(cfg.events.object_visual)
        self.assertIsNone(cfg.events.lighting)
        self.assertEqual(
            cfg.events.reset_layout.params["object_xy_range_full"], (0.0, 0.0)
        )
        self.assertFalse(cfg.observations.wrist.rgb.params["randomize"])


if __name__ == "__main__":
    unittest.main()
