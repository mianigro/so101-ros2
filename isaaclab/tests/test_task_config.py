"""Configuration contract tests that do not launch Isaac Sim."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import gymnasium as gym

from so101_rl.camera_profile import (
    POLICY_FREQUENCY_HZ,
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
)
from so101_rl.export_manifest import validate_deployable_contract
from so101_rl.tasks.common import SO101VisualObservationsCfg
from so101_rl.tasks.common.visual_env_cfg import (
    SO101_FIXED_JAW_PAD_PRIM_PATH,
    SO101_MOVING_JAW_PAD_PRIM_PATH,
)
from so101_rl.tasks.common.agents.rsl_rl_ppo_cfg import SO101VisualPPOCfg
from so101_rl.tasks.object_in_cup.agents.rsl_rl_vision_ppo_cfg import (
    SO101ObjectInCupVisionPPOCfg,
)
from so101_rl.tasks.object_in_cup.mdp.critic_observations import CRITIC_STATE_DIM
from so101_rl.tasks.object_in_cup.object_in_cup_env_cfg import (
    SO101ObjectInCupVisionEnvCfg,
    SO101ObjectInCupVisionFixedEnvCfg,
)
from so101_rl.tasks.three_boxes_in_cups.agents.rsl_rl_vision_ppo_cfg import (
    SO101ThreeBoxesInCupsVisionPPOCfg,
)
from so101_rl.tasks.three_boxes_in_cups.mdp.critic_observations import (
    THREE_BOX_CRITIC_STATE_DIM,
)
from so101_rl.tasks.three_boxes_in_cups.three_boxes_in_cups_env_cfg import (
    BOX_NAMES,
    BOX_STARTS,
    CUP_NAMES,
    CUP_STARTS,
    SO101ThreeBoxesInCupsVisionEnvCfg,
    SO101ThreeBoxesInCupsVisionFixedEnvCfg,
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

    def _three_box_config(self, fixed: bool = False):
        config_type = (
            SO101ThreeBoxesInCupsVisionFixedEnvCfg
            if fixed
            else SO101ThreeBoxesInCupsVisionEnvCfg
        )
        with patch(
            "so101_rl.tasks.three_boxes_in_cups.three_boxes_in_cups_env_cfg.load_asset_manifest",
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
                "SO101-Three-Boxes-In-Cups-Vision-Fixed-v0",
                "SO101-Three-Boxes-In-Cups-Vision-v0",
            },
        )

    def test_control_rate_action_contract_and_geometry_binding(self):
        cfg = self._config()
        cfg.validate()
        agent = SO101ObjectInCupVisionPPOCfg()

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
        self.assertEqual(agent.clip_actions, 1.0)
        self.assertEqual(cfg.terminations.success.params["required_steps"], 15)
        self.assertEqual(cfg.events.reset_layout.params["curriculum_steps"], 45_000_000)
        self.assertEqual(cfg.terminations.success.params["xy_tolerance"], 0.0038)
        self.assertEqual(
            tuple(cfg.rewards.__dataclass_fields__)[:5],
            (
                "approach_progress",
                "closure_progress",
                "grasp_acquired",
                "grasp_held",
                "lift_progress",
            ),
        )
        self.assertEqual(cfg.rewards.approach_progress.weight, 1.0)
        self.assertEqual(cfg.rewards.closure_progress.weight, 0.1)
        self.assertEqual(
            cfg.rewards.closure_progress.params["minimum_insertion"], 0.001
        )
        self.assertEqual(
            cfg.rewards.closure_progress.params["fixed_pad_length"], 0.025
        )
        self.assertEqual(cfg.rewards.grasp_acquired.weight, 1.0)
        self.assertEqual(cfg.rewards.grasp_held.weight, 0.5)
        self.assertEqual(cfg.rewards.grasp_held.params["minimum_insertion"], 0.001)
        self.assertEqual(cfg.rewards.lift_progress.weight, 0.2)
        self.assertEqual(cfg.rewards.lift_progress.params["lift_clearance"], 0.001)
        self.assertNotIn("lift_height", cfg.rewards.lift_progress.params)
        self.assertAlmostEqual(
            cfg.rewards.transport.params["minimum_height"], 0.0135
        )
        self.assertEqual(
            cfg.rewards.approach_progress.params["half_extents"],
            (0.0125, 0.0125, 0.0125),
        )
        self.assertEqual(cfg.scene.fixed_jaw_contact.history_length, 4)
        self.assertEqual(cfg.scene.fixed_jaw_contact.update_period, 0.0)
        self.assertTrue(cfg.scene.fixed_jaw_contact.track_pose)
        self.assertEqual(cfg.scene.fixed_jaw_contact.force_threshold, 0.1)
        self.assertEqual(
            cfg.scene.fixed_jaw_contact.prim_path, SO101_FIXED_JAW_PAD_PRIM_PATH
        )
        self.assertEqual(
            cfg.scene.moving_jaw_contact.prim_path, SO101_MOVING_JAW_PAD_PRIM_PATH
        )
        self.assertTrue(cfg.scene.moving_jaw_contact.track_pose)
        self.assertEqual(
            cfg.scene.ee_frame.target_frames[0].offset.pos,
            (0.0052, -0.000218, -0.0925),
        )

    def test_visual_actor_and_simulator_critic_contract(self):
        cfg = self._config()
        cfg.validate()
        agent = SO101ObjectInCupVisionPPOCfg()
        common_agent = SO101VisualPPOCfg()

        common_observations = SO101VisualObservationsCfg()
        self.assertEqual(
            tuple(common_observations.__dataclass_fields__),
            SO101_ACTOR_OBSERVATION_GROUPS,
        )

        self.assertEqual(
            tuple(agent.obs_groups["actor"]), SO101_ACTOR_OBSERVATION_GROUPS
        )
        self.assertEqual(agent.actor.class_name, common_agent.actor.class_name)
        self.assertEqual(agent.actor.hidden_dims, common_agent.actor.hidden_dims)
        self.assertEqual(agent.actor.cnn_cfg, common_agent.actor.cnn_cfg)
        self.assertEqual(agent.critic.hidden_dims, common_agent.critic.hidden_dims)
        self.assertEqual(
            agent.algorithm.class_name, common_agent.algorithm.class_name
        )
        self.assertEqual(agent.obs_groups["critic"], ["critic_state"])
        self.assertFalse(cfg.observations.critic_state.enable_corruption)
        self.assertEqual(CRITIC_STATE_DIM, 34)
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
        self.assertEqual(agent.num_steps_per_env, 48)
        self.assertEqual(agent.algorithm.gamma, 0.9933)
        self.assertEqual(agent.algorithm.lam, 0.9664)
        self.assertEqual(agent.algorithm.learning_rate, 7.0e-5)
        self.assertEqual(agent.algorithm.schedule, "fixed")
        self.assertEqual(agent.algorithm.num_mini_batches, 8)
        self.assertEqual(agent.actor.hidden_dims, [512, 256, 128])
        self.assertEqual(agent.critic.hidden_dims, [256, 256, 128])
        self.assertEqual(agent.critic.class_name, "MLPModel")
        self.assertIsNone(agent.critic.distribution_cfg)

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

    def test_deployment_validation_uses_contract_not_task_allowlist(self):
        agent = SO101ObjectInCupVisionPPOCfg()
        validate_deployable_contract(
            "SO101-Object-In-Cup-Vision-v0", self._config(), agent
        )
        validate_deployable_contract(
            "SO101-Object-In-Cup-Vision-Fixed-v0", self._config(fixed=True), agent
        )

        incompatible = self._config()
        incompatible.actions.joint_delta.scale["gripper"] = 0.20
        with self.assertRaisesRegex(ValueError, "action scale"):
            validate_deployable_contract("SO101-New-Scenario-Vision-v0", incompatible, agent)

    def test_three_box_scenario_composes_the_shared_platform(self):
        cfg = self._three_box_config()
        cfg.validate()
        agent = SO101ThreeBoxesInCupsVisionPPOCfg()

        self.assertEqual(cfg.episode_length_s, 45.0)
        self.assertEqual(THREE_BOX_CRITIC_STATE_DIM, 84)
        self.assertEqual(
            tuple(agent.obs_groups["actor"]), SO101_ACTOR_OBSERVATION_GROUPS
        )
        self.assertEqual(agent.obs_groups["critic"], ["critic_state"])
        self.assertFalse(cfg.observations.critic_state.enable_corruption)
        self.assertEqual(agent.actor.hidden_dims, [512, 256, 128])
        self.assertEqual(agent.critic.hidden_dims, [256, 256, 128])
        for name in (*BOX_NAMES, *CUP_NAMES):
            self.assertTrue(hasattr(cfg.scene, name))
        self.assertEqual(
            cfg.scene.fixed_jaw_contact.filter_prim_paths_expr,
            [
                "{ENV_REGEX_NS}/Box1",
                "{ENV_REGEX_NS}/Box2",
                "{ENV_REGEX_NS}/Box3",
            ],
        )
        self.assertEqual(cfg.rewards.closure_progress.weight, 0.1)
        self.assertEqual(
            cfg.rewards.closure_progress.params["minimum_insertion"], 0.001
        )
        self.assertEqual(
            cfg.rewards.closure_progress.params["fixed_pad_length"], 0.025
        )
        self.assertTrue(cfg.scene.moving_jaw_contact.track_pose)
        self.assertEqual(cfg.rewards.grasp_acquired.weight, 1.0)
        self.assertEqual(cfg.rewards.grasp_held.weight, 0.5)
        self.assertEqual(cfg.rewards.lift_progress.weight, 0.2)
        self.assertEqual(cfg.rewards.lift_progress.params["lift_clearance"], 0.001)
        self.assertNotIn("lift_height", cfg.rewards.lift_progress.params)
        self.assertAlmostEqual(
            cfg.rewards.transport.params["minimum_height"], 0.0135
        )
        validate_deployable_contract(
            "SO101-Three-Boxes-In-Cups-Vision-v0", cfg, agent
        )

    def test_three_box_fixed_variant_disables_shared_and_task_variation(self):
        cfg = self._three_box_config(fixed=True)
        cfg.validate()

        self.assertEqual(cfg.actions.joint_delta.max_delay_steps, 0)
        self.assertIsNone(cfg.events.randomize_cameras)
        self.assertIsNone(cfg.events.actuator_response)
        self.assertIsNone(cfg.events.table_visual)
        self.assertTrue(cfg.events.reset_layout.params["fixed_layout"])
        for index, name in enumerate(BOX_NAMES):
            self.assertEqual(
                getattr(cfg.scene, name).init_state.pos, BOX_STARTS[index]
            )
        for index, name in enumerate(CUP_NAMES):
            self.assertEqual(
                getattr(cfg.scene, name).init_state.pos, CUP_STARTS[index]
            )
        for name in BOX_NAMES:
            self.assertIsNone(getattr(cfg.events, f"{name}_material"))
            self.assertIsNone(getattr(cfg.events, f"{name}_mass"))
            self.assertIsNone(getattr(cfg.events, f"{name}_visual"))
        for name in CUP_NAMES:
            self.assertIsNone(getattr(cfg.events, f"{name}_material"))
            self.assertIsNone(getattr(cfg.events, f"{name}_visual"))
        validate_deployable_contract(
            "SO101-Three-Boxes-In-Cups-Vision-Fixed-v0",
            cfg,
            SO101ThreeBoxesInCupsVisionPPOCfg(),
        )


if __name__ == "__main__":
    unittest.main()
