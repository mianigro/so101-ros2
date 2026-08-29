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

"""End-to-end serving smoke (no robot, needs CUDA + the local dataset).

Exercises the exact server path: serving checkpoint (feature-renamed config
+ stage-stats processors) -> ``InferenceEngine.load_policy`` with
``policy_type="pi05_icl"`` (the SUPPORTED_POLICIES gate + strict
``from_pretrained`` + feature validation + processor load) ->
``predict_action_chunk`` through the engine's pre/post processors, with and
without a demo pack on the policy.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

LOCAL_DATASET = Path.home() / ".cache/huggingface/lerobot/local/so101_test/meta/info.json"


def _client_features():
    """Hand-built equivalent of build_lerobot_features('monomanual_dual_overhead')."""
    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": None}}
    for cam in ("wrist", "overhead_1", "overhead_2"):
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
        }
    return features


def _raw_obs(device):
    g = torch.Generator().manual_seed(11)
    obs = {
        f"observation.images.{cam}": torch.rand(1, 3, 480, 640, generator=g).to(device)
        for cam in ("wrist", "overhead_1", "overhead_2")
    }
    obs["observation.state"] = (torch.rand(1, 6, generator=g) * 2 - 1).to(device)
    obs["task"] = ["pick up the cube and place it in the container."]
    return obs


@unittest.skipUnless(
    torch.cuda.is_available() and LOCAL_DATASET.exists(),
    "needs CUDA and the local so101_test dataset",
)
class TestServingE2E(unittest.TestCase):
    def test_engine_loads_and_predicts_pi05_icl(self):
        import so101_icl.registration  # noqa: F401

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from so101_icl import PI05_BASE_IMAGE_KEY_MAP
        from so101_icl.lora import setup_trainable_policy
        from so101_icl.modeling_pi05_icl import PI05ICLPolicy, save_serving_checkpoint

        policy = PI05ICLPolicy.from_base(device="cuda", dtype="bfloat16")
        setup_trainable_policy(policy, policy.config)
        stats = LeRobotDataset("local/so101_test").meta.stats
        # serving config speaks the ROS setup's camera names (inverse rename)
        canonical_to_setup = {v: k for k, v in PI05_BASE_IMAGE_KEY_MAP.items()}

        with tempfile.TemporaryDirectory(prefix="icl_serving_") as tmp:
            ckpt = save_serving_checkpoint(
                policy, tmp, input_feature_rename=canonical_to_setup, dataset_stats=stats
            )
            self.assertTrue((ckpt / "model.safetensors").exists())
            self.assertTrue((ckpt / "policy_preprocessor.json").exists())
            del policy
            torch.cuda.empty_cache()

            from lerobot.async_inference.helpers import RemotePolicyConfig
            from policy_server.inference_engine import InferenceEngine

            engine = InferenceEngine()
            engine.load_policy(
                RemotePolicyConfig(
                    policy_type="pi05_icl",
                    pretrained_name_or_path=str(ckpt),
                    lerobot_features=_client_features(),
                    actions_per_chunk=16,
                    device="cuda",
                )
            )
            self.assertEqual(engine.policy_type, "pi05_icl")

            obs = _raw_obs("cuda")
            processed = engine.preprocessor(obs)
            with torch.no_grad():
                chunk = engine.policy.predict_action_chunk(processed)
            # postprocess per step, exactly like InferenceEngine._predict_action_chunk.
            # The policy emits its full chunk_size; the async loop slices to
            # actions_per_chunk downstream.
            steps = [engine.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])]
            out = torch.stack(steps, dim=1).squeeze(0)
            self.assertEqual(tuple(out.shape), (engine.policy.config.chunk_size, 6))
            self.assertEqual(engine.actions_per_chunk, 16)
            self.assertTrue(torch.isfinite(out).all())

            # demo side channel against the ENGINE's policy
            from so101_icl import demo_transport

            demo_transport._current_policy["policy"] = engine.policy
            try:
                de = engine.policy.config.demo_encoder
                pack = (
                    torch.rand(1, de.k_max, de.frames_per_demo, 3, 224, 224, device="cuda") * 2 - 1,
                    torch.tensor([[True, True, False, False]], device="cuda"),
                    torch.randn(1, de.k_max, de.traj_steps, 64, device="cuda"),
                    torch.tensor([[1.0, 1.0, 0.0, 0.0]], device="cuda"),
                )
                engine.policy.model.set_demo_pack(*pack)
                torch.manual_seed(3)
                with torch.no_grad():
                    chunk_icl = engine.policy.predict_action_chunk(processed)
                self.assertTrue(torch.isfinite(chunk_icl).all())
                engine.policy.model.clear_demo_pack()
            finally:
                demo_transport._current_policy["policy"] = None
            engine.unload_policy()


if __name__ == "__main__":
    unittest.main()
