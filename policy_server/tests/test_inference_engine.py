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

"""Tests for policy-server feature compatibility checks."""

from types import SimpleNamespace

import pytest

from lerobot.async_inference.helpers import RemotePolicyConfig
from policy_server.grpc_server import PolicyServer
from policy_server.inference_engine import InferenceEngine, validate_policy_input_features


def _policy_features(*camera_names: str, state_shape: tuple[int, ...] = (6,)):
    features = {"observation.state": SimpleNamespace(shape=state_shape)}
    features.update(
        {
            f"observation.images.{name}": SimpleNamespace(shape=(3, 480, 640))
            for name in camera_names
        }
    )
    return features


def _client_features(*camera_names: str):
    features = {"observation.state": {"shape": (6,)}}
    features.update(
        {
            f"observation.images.{name}": {"shape": (480, 640, 3)}
            for name in camera_names
        }
    )
    return features


def test_policy_server_accepts_exact_monomanual_dual_overhead_schema():
    validate_policy_input_features(
        _policy_features("wrist", "overhead_1", "overhead_2"),
        _client_features("wrist", "overhead_1", "overhead_2"),
    )


def test_grpc_server_enforces_fixed_30hz_config():
    config = SimpleNamespace(fps=30, inference_latency=0.033, obs_queue_timeout=2.0)
    server = PolicyServer(config)
    assert server.engine.config.fps == 30

    config.fps = 50
    with pytest.raises(ValueError, match="frequency must be 30 Hz"):
        PolicyServer(config)


def test_policy_server_rejects_noncanonical_or_missing_camera_keys():
    with pytest.raises(ValueError, match="missing=.*overhead_2.*unexpected=.*unexpected"):
        validate_policy_input_features(
            _policy_features("wrist", "overhead_1", "unexpected"),
            _client_features("wrist", "overhead_1", "overhead_2"),
        )


def test_policy_server_rejects_incompatible_state_shape():
    with pytest.raises(ValueError, match=r"must be \(6,\), got \(7,\)"):
        validate_policy_input_features(
            _policy_features("wrist", "overhead_1", state_shape=(7,)),
            _client_features("wrist", "overhead_1"),
        )


def test_policy_server_rejects_client_camera_rename_map_before_loading():
    config = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="not-loaded",
        lerobot_features=_client_features("wrist", "overhead_1"),
        actions_per_chunk=1,
        rename_map={"observation.images.unexpected": "observation.images.overhead_1"},
    )

    with pytest.raises(ValueError, match="rename_map is not supported"):
        InferenceEngine().load_policy(config)


def test_load_policy_validates_schema_before_building_processors(monkeypatch):
    class _FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(
                input_features=_policy_features("wrist", "unexpected")
            )

        def to(self, device):
            raise AssertionError("incompatible policy must not be moved to a device")

    class _FakePolicyClass:
        @staticmethod
        def from_pretrained(repo_id):
            return _FakePolicy()

    monkeypatch.setattr(
        "policy_server.inference_engine.get_policy_class",
        lambda policy_type: _FakePolicyClass,
    )
    config = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="fake-policy",
        lerobot_features=_client_features("wrist", "overhead_1"),
        actions_per_chunk=1,
    )

    with pytest.raises(ValueError, match="missing=.*overhead_1.*unexpected=.*unexpected"):
        InferenceEngine().load_policy(config)
