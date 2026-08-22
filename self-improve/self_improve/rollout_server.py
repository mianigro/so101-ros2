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

"""Rollout policy server: keeps a LeRobot policy on the GPU and serves batched
action chunks to the Isaac Sim rollout client (or any client speaking
``self_improve.wire``).

This deliberately mirrors the proven load/inference path of
``policy_server/inference_engine.py`` (``get_policy_class`` -> ``from_pretrained``
-> ``make_pre_post_processors`` -> ``predict_action_chunk`` -> postprocess),
slimmed down to a synchronous whole-batch API: the sim client sends every
environment needing a chunk refill in one request, so one PaliGemma forward
covers them all.  Batches whose pipeline rejects multi-item inputs fall back
to a per-item loop automatically (see ``--batch-inference``).
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

from . import contract, wire

logger = logging.getLogger(__name__)

SUPPORTED_POLICIES = ("pi05", "pi0", "smolvla")


class RolloutPolicyServer:
    """Loads one policy and answers wire requests on a single connection."""

    def __init__(self, repo_id: str, *, policy_type: str = "pi05",
                 device: str = "cuda", actions_per_chunk: int = 16,
                 batch_inference: str = "auto",
                 dtype: str = "bfloat16") -> None:
        if policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"unsupported policy type {policy_type!r}; "
                f"expected one of {SUPPORTED_POLICIES}"
            )
        if batch_inference not in ("auto", "on", "off"):
            raise ValueError("batch_inference must be 'auto', 'on', or 'off'")
        if dtype not in ("auto", "bfloat16", "float32"):
            raise ValueError("dtype must be 'auto', 'bfloat16', or 'float32'")
        self.repo_id = repo_id
        self.policy_type = policy_type
        self.device = device
        self.actions_per_chunk = int(actions_per_chunk)
        self.batch_inference = batch_inference
        self.dtype = dtype
        self._batched_ok: Optional[bool] = None  # resolved on first batch

        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.client_features: Optional[Dict[str, dict]] = None
        # Zero-shot adaptation for base checkpoints whose pretraining schema
        # differs from the rig's (openpi camera keys, padded 32-dim spaces):
        self._image_key_map: Dict[str, str] = {}
        self._policy_state_dim: Optional[int] = None
        self._client_action_dim: Optional[int] = None
        self._model_dtype: Optional[torch.dtype] = None

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        start = time.perf_counter()
        from lerobot.configs.policies import PreTrainedConfig

        policy_class = get_policy_class(self.policy_type)
        # pi05's constructor moves the model to ``config.device`` at build
        # time, and published checkpoints store devices from other machines
        # (lerobot/pi05_base says "mps", which lerobot silently remaps to
        # cuda:0). Left alone, from_pretrained drops the full fp32 model
        # (~15 GiB) onto GPU 0 before --device/--dtype apply — instant OOM on
        # a 16 GB card. Build on the CPU instead; cast and move below.
        policy_config = PreTrainedConfig.from_pretrained(self.repo_id)
        policy_config.device = "cpu"
        policy = policy_class.from_pretrained(
            self.repo_id, config=policy_config)
        if self.dtype == "bfloat16":
            # bf16 is the standard pi0.5 inference precision; cast on the
            # CPU so the GPU never holds the fp32 copy.
            policy.to(dtype=torch.bfloat16)
        elif self.dtype == "float32":
            policy.to(dtype=torch.float32)
        policy.to(self.device)
        policy.config.device = self.device
        policy.eval()
        device_override = {"device": self.device}
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=self.repo_id,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        # Processors run fp32; the model may be half precision (see --dtype).
        self._model_dtype = next(
            (p.dtype for p in policy.parameters() if p.is_floating_point()),
            None)
        memory_note = ""
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            memory_note = (
                f", {torch.cuda.memory_allocated(self.device) / 2**30:.1f} GiB"
                f" allocated")
        logger.info(
            "policy %s (%s) loaded on %s in %.1fs (dtype=%s%s)",
            self.repo_id, self.policy_type, self.device,
            time.perf_counter() - start,
            next(p.dtype for p in policy.parameters() if p.is_floating_point()),
            memory_note,
        )

    # ------------------------------------------------------------------
    # Wire handling
    # ------------------------------------------------------------------

    def handle_connection(self, sock: Any) -> None:
        """Serve one client connection until it says bye or drops."""
        self._batched_ok = None if self.batch_inference == "auto" \
            else self.batch_inference == "on"
        while True:
            message = wire.recv_msg(sock)
            if not isinstance(message, dict):
                raise wire.WireError("expected dict message")
            msg_type = message.get("type")
            if msg_type == "bye":
                return
            if msg_type == "hello":
                response = self._handle_hello(message)
            elif msg_type == "infer":
                response = self._handle_infer(message)
            else:
                response = {"type": "error",
                            "message": f"unknown message type {msg_type!r}"}
            wire.send_msg(sock, response)

    def _handle_hello(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = message["features"]
            image_key_map = dict(message.get("image_key_map") or {})
            client_action_dim = message.get("action_dim")
            requested_chunk = int(message.get(
                "actions_per_chunk", self.actions_per_chunk))
            self._validate_features(features, image_key_map)
            self.actions_per_chunk = max(1, requested_chunk)
            self.client_features = features
            self._image_key_map = image_key_map
            self._client_action_dim = (
                int(client_action_dim) if client_action_dim
                else self._action_dim())
            logger.info(
                "handshake ok: features=%s chunk=%d action_dim=%d "
                "(policy pads state to %s)",
                sorted(features), self.actions_per_chunk,
                self._client_action_dim, self._policy_state_dim,
            )
            return {"type": "welcome", "chunk": self.actions_per_chunk,
                    "action_dim": self._client_action_dim}
        except Exception as exc:
            logger.exception("handshake rejected")
            return {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    def _validate_features(self, features: Dict[str, dict],
                           image_key_map: Dict[str, str]) -> None:
        policy_inputs = self.policy.config.input_features
        if not policy_inputs:
            raise ValueError("loaded policy does not declare input_features")
        client_images = {k for k in features
                         if k.startswith("observation.images.")}
        mapped_images = {image_key_map.get(k, k) for k in client_images}
        policy_images = {k for k in policy_inputs
                         if k.startswith("observation.images.")}
        if policy_images != mapped_images:
            raise ValueError(
                f"camera schema mismatch (after remap); missing="
                f"{sorted(mapped_images - policy_images)}, unexpected="
                f"{sorted(policy_images - mapped_images)}; pass "
                f"image_key_map to adapt a zero-shot base checkpoint"
            )
        client_state = features.get(OBS_STATE, {})
        client_shape = tuple(client_state.get("shape", ()))
        state_shape = tuple(getattr(policy_inputs.get(OBS_STATE), "shape", ()))
        if len(client_shape) != 1 or len(state_shape) != 1:
            raise ValueError(
                f"state shapes must be vectors: client {client_shape}, "
                f"policy {state_shape}")
        if client_shape[0] > state_shape[0]:
            raise ValueError(
                f"client state dim {client_shape[0]} exceeds policy "
                f"{state_shape[0]}")
        self._policy_state_dim = state_shape[0]

    def _action_dim(self) -> int:
        output = self.policy.config.output_features
        action_feature = output.get(contract.ACTION_FEATURE_KEY)
        shape = tuple(getattr(action_feature, "shape", ()))
        if len(shape) != 1:
            raise ValueError(f"unexpected action feature shape {shape}")
        return int(shape[0])

    def _handle_infer(self, message: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            images = message["images"]
            state = np.asarray(message["state"], dtype=np.float32)
            task = str(message["task"])
            batch = int(state.shape[0])
            actions = self._predict(images, state, task, batch)
            elapsed = time.perf_counter() - start
            logger.info("infer batch=%d chunk=%d in %.2fs (%.0f ms/env)",
                        batch, actions.shape[1], elapsed,
                        1000.0 * elapsed / max(batch, 1))
            return {"type": "chunk", "actions": actions}
        except Exception as exc:
            logger.exception("inference failed")
            return {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _observation(self, images: Dict[str, np.ndarray], state: np.ndarray,
                     task: str, batch: int) -> Dict[str, torch.Tensor | str]:
        state_tensor = torch.tensor(np.asarray(state), dtype=torch.float32)
        if (self._policy_state_dim is not None
                and state_tensor.shape[-1] < self._policy_state_dim):
            # Base checkpoints train with zero-padded state/action spaces;
            # pad the rig's 6 joints the same way.
            padding = torch.zeros(
                batch, self._policy_state_dim - state_tensor.shape[-1])
            state_tensor = torch.cat((state_tensor, padding), dim=-1)
        observation: Dict[str, Any] = {
            OBS_STATE: state_tensor,
            "task": task,
        }
        for key, array in images.items():
            tensor = torch.tensor(np.asarray(array))
            tensor = tensor.permute(0, 3, 1, 2).to(  # BHWC -> BCHW
                dtype=torch.float32).div_(255).contiguous()
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"image batch {tensor.shape[0]} != state batch {batch}"
                )
            observation[self._image_key_map.get(key, key)] = tensor
        return observation

    def _predict_chunk_autocast(self, observation: Dict[str, Any]):
        """Run chunk prediction under bf16 autocast when the model is half.

        lerobot's flow-matching internals sample noise and timesteps in
        float32 regardless of model precision (flow_matching.sample_noise),
        so a bf16-cast model hits "mat1 and mat2 must have the same dtype"
        at the action expert unless matmul inputs are auto-cast.
        """
        half = (self._model_dtype is not None
                and self._model_dtype != torch.float32)
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type,
                            dtype=self._model_dtype or torch.bfloat16,
                            enabled=half):
            return self.policy.predict_action_chunk(observation)

    def _predict(self, images: Dict[str, np.ndarray], state: np.ndarray,
                 task: str, batch: int) -> np.ndarray:
        if self._batched_ok is not False:
            try:
                return self._predict_batched(images, state, task, batch)
            except Exception as exc:
                if self.batch_inference == "on":
                    raise
                logger.warning(
                    "batched inference failed (%s: %s); falling back to "
                    "per-environment inference", type(exc).__name__, exc,
                )
                self._batched_ok = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self._predict_per_env(images, state, task, batch)

    def _predict_batched(self, images: Dict[str, np.ndarray], state: np.ndarray,
                         task: str, batch: int) -> np.ndarray:
        observation = self._observation(images, state, task, batch)
        chunk = self._run_policy(observation)
        if self._batched_ok is None:
            self._batched_ok = True
        return chunk

    def _predict_per_env(self, images: Dict[str, np.ndarray], state: np.ndarray,
                         task: str, batch: int) -> np.ndarray:
        chunks = []
        for index in range(batch):
            single_images = {key: value[index:index + 1]
                             for key, value in images.items()}
            observation = self._observation(
                single_images, state[index:index + 1], task, 1,
            )
            chunks.append(self._run_policy(observation))
        return np.concatenate(chunks, axis=0)

    def _run_policy(self, observation: Dict[str, Any]) -> np.ndarray:
        """Preprocess -> chunk prediction -> postprocess, on the GPU."""
        observation = self.preprocessor(observation)
        # Feed a half-precision model bf16 inputs; un-normalize in fp32.
        if (self._model_dtype is not None
                and self._model_dtype != torch.float32):
            observation = {
                key: (value.to(self._model_dtype)
                      if torch.is_tensor(value) and value.is_floating_point()
                      else value)
                for key, value in observation.items()
            }
        chunk = self._predict_chunk_autocast(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :].float()
        processed = []
        for index in range(chunk.shape[1]):
            processed.append(self.postprocessor(chunk[:, index]))
        stacked = torch.stack(processed, dim=1)
        if (self._client_action_dim is not None
                and stacked.shape[-1] > self._client_action_dim):
            # Drop the zero-padding dims the rig never uses.
            stacked = stacked[..., : self._client_action_dim]
        return stacked.detach().cpu().numpy().astype(np.float32)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a LeRobot policy for autonomous rollouts.",
    )
    parser.add_argument("--repo-id", required=True,
                        help="checkpoint path or HF repo id of the policy")
    parser.add_argument("--policy-type", default="pi05",
                        choices=SUPPORTED_POLICIES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default=wire.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=wire.DEFAULT_PORT)
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--batch-inference", default="auto",
                        choices=("auto", "on", "off"),
                        help="auto: try batched forward, fall back per-env")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("auto", "bfloat16", "float32"),
        help="model precision at inference. bfloat16 (default) halves the "
             "3B footprint — published pi05_base stores float32 (~12.4 GB "
             "weights) which OOMs 16 GB cards once sim batches are added. "
             "auto keeps the checkpoint's stored dtype.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = RolloutPolicyServer(
        args.repo_id,
        policy_type=args.policy_type,
        device=args.device,
        actions_per_chunk=args.actions_per_chunk,
        batch_inference=args.batch_inference,
        dtype=args.dtype,
    )
    server.load()
    wire.serve_forever(args.host, args.port, server.handle_connection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
