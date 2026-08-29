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

"""``pi05_icl`` model: PI05 with in-context demo tokens (ICL §4.1).

``PI05ICLCore`` overrides exactly one hot path — ``embed_prefix`` — to splice
cached demo tokens between the camera and language prefix blocks. Because
both the training ``forward`` and the inference ``sample_actions`` go through
``embed_prefix``, and ``position_ids`` / KV-cache ``prefix_offsets`` derive
from the returned ``pad_masks``, everything downstream adapts automatically.

Loading gotcha (ICL §4.0.1): ``PI05Policy.from_pretrained`` defaults to
``strict=True`` and swallows load failures with a printed warning. Our
checkpoints legitimately contain keys missing from ``lerobot/pi05_base``
(``demo_encoder``, gates), so every base load MUST pass ``strict=False`` and
should be followed by :func:`assert_weights_loaded`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
)
from lerobot.utils.import_utils import require_package

from .configuration_pi05_icl import BASE_CHECKPOINT, ICLConfig
from .demo_encoder import DemoEncoder

logger = logging.getLogger(__name__)

# Batch keys produced by so101_icl.data.ICLDataset collation (training path).
DEMO_FRAMES = "icl.demo_frames"        # [B, k_max, F, 3, H, W] preprocessed
DEMO_MASK = "icl.demo_mask"            # [B, k_max] bool
DEMO_TRAJ = "icl.demo_traj"            # [B, k_max, S, d_state + d_action] normalized
DEMO_TRAJ_OK = "icl.demo_traj_ok"      # [B, k_max] float 0/1


class PI05ICLCore(PI05Pytorch):
    """PI05Pytorch with a DemoEncoder and a cached demo-token pack."""

    def __init__(self, config: ICLConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        vlm_width = get_gemma_config(config.paligemma_variant).width
        self.demo_encoder = DemoEncoder(config, vlm_width)
        # (embs [B0, T, D], pad [B0, T]) or None; B0 == 1 for inference packs,
        # == batch size (or 1, expanded later) for training packs.
        self._demo_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        # Extra multiplicative factor on the demo gates; set to 0.0 to compute
        # loss_demo_zeroed without touching the learned gate parameters.
        self._demo_gate_scale: float = 1.0
        self._assert_vlm_attention_path()

    def _assert_vlm_attention_path(self) -> None:
        """Fail loudly if a lerobot bump moved the LoRA target modules."""
        try:
            lm = self.paligemma_with_expert.paligemma.model.language_model
            _ = lm.layers[0].self_attn.q_proj
        except AttributeError as e:
            raise RuntimeError(
                "pi05_icl: the VLM attention module path changed in this "
                "lerobot version ('paligemma_with_expert.paligemma.model."
                "language_model.layers[i].self_attn.*_proj'). Update "
                "so101_icl/lora.py targets and this check."
            ) from e

    # ------------------------------------------------------------------ #
    # Demo-pack lifecycle (ICL §2.4)                                     #
    # ------------------------------------------------------------------ #

    def set_demo_pack(
        self,
        frames: torch.Tensor,
        demo_mask: torch.Tensor,
        traj: torch.Tensor,
        traj_ok: torch.Tensor,
    ) -> None:
        """Encode a demo pack ONCE (no grad) and cache it for inference.

        All subtask ticks then reuse the cached tokens inside
        ``embed_prefix``; only the extra prefix attention columns are paid
        per chunk query (KV cache is rebuilt per query either way).
        """
        embs, pad = self.demo_encoder.encode_pack(
            frames, demo_mask, traj, traj_ok, self._embed_demo_frames
        )
        self._demo_cache = (embs, pad)

    def set_demo_pack_train(
        self,
        frames: torch.Tensor,
        demo_mask: torch.Tensor,
        traj: torch.Tensor,
        traj_ok: torch.Tensor,
    ) -> None:
        """Training path: run the encoder inside the autograd graph.

        The cached tensors stay attached to the graph so the spliced demo
        tokens propagate gradients into the DemoEncoder / gates when the
        flow-matching loss is backwarded.
        """
        embs, pad = self.demo_encoder(
            frames, demo_mask, traj, traj_ok, self._embed_demo_frames,
            gate_scale=self._demo_gate_scale,
        )
        self._demo_cache = (embs, pad)

    def clear_demo_pack(self) -> None:
        self._demo_cache = None

    @property
    def demo_pack_is_set(self) -> bool:
        return self._demo_cache is not None

    def _embed_demo_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self.paligemma_with_expert.embed_image(frames)

    # ------------------------------------------------------------------ #
    # The single injection point                                         #
    # ------------------------------------------------------------------ #

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Base prefix with demo tokens spliced in: [cams | demo | language].

        Demo tokens get ``att=0`` (bidirectional prefix block); their pad
        mask entries drive ``position_ids`` and the KV-cache
        ``prefix_offsets`` downstream, so nothing else needs to change.
        """
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        if self._demo_cache is None:
            return embs, pad_masks, att_masks
        d_embs, d_pad = self._demo_cache
        return splice_demo_tokens(embs, pad_masks, att_masks, d_embs, d_pad, masks.shape[1])


def splice_demo_tokens(
    embs: torch.Tensor,
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    d_embs: torch.Tensor,
    d_pad: torch.Tensor,
    n_lang_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert demo tokens between the camera and language prefix blocks.

    Pure tensor logic, extracted for unit testing. Demo tokens get ``att=0``
    (bidirectional prefix block); their ``pad_masks`` entries drive
    ``position_ids`` and the KV-cache ``prefix_offsets`` downstream, so
    nothing outside the prefix needs to change. A single-sample cache
    (batch 1) is expanded to the observation batch.
    """
    batch = embs.shape[0]
    if d_embs.shape[0] == 1 and batch > 1:
        d_embs = d_embs.expand(batch, -1, -1)
        d_pad = d_pad.expand(batch, -1)
    elif d_embs.shape[0] != batch:
        raise ValueError(
            f"demo cache batch {d_embs.shape[0]} does not match observation batch {batch}"
        )
    d_embs = d_embs.to(dtype=embs.dtype)

    n_img = embs.shape[1] - n_lang_tokens  # everything before the language block
    n_demo = d_embs.shape[1]
    zeros = torch.zeros(
        (att_masks.shape[0], n_demo), dtype=att_masks.dtype, device=att_masks.device
    )
    embs = torch.cat([embs[:, :n_img], d_embs, embs[:, n_img:]], dim=1)
    pad_masks = torch.cat([pad_masks[:, :n_img], d_pad, pad_masks[:, n_img:]], dim=1)
    att_masks = torch.cat([att_masks[:, :n_img], zeros, att_masks[:, n_img:]], dim=1)
    return embs, pad_masks, att_masks


class PI05ICLPolicy(PI05Policy):
    """pi05_icl policy wrapper; everything except the core model and the
    training-forward demo plumbing is inherited unchanged."""

    config_class = ICLConfig
    name = "pi05_icl"

    def __init__(self, config: ICLConfig, **kwargs):
        require_package("transformers", extra="pi")
        # Mirrors PI05Policy.__init__ but builds PI05ICLCore.
        from lerobot.policies.pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05ICLCore(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def forward(self, batch: dict[str, torch.Tensor], reduction: str = "mean"):
        """Training forward: refresh the demo pack from the batch first.

        Batches without demo keys (bare-prompt / demo-zeroed evals) run with
        the pack cleared, i.e. exactly the pi05 base behavior.
        """
        model: PI05ICLCore = self.model
        if DEMO_FRAMES in batch:
            model.set_demo_pack_train(
                batch[DEMO_FRAMES], batch[DEMO_MASK], batch[DEMO_TRAJ], batch[DEMO_TRAJ_OK]
            )
        else:
            model.clear_demo_pack()
        return super().forward(batch, reduction=reduction)

    # ------------------------------------------------------------------ #
    # Load helpers                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_base(
        cls,
        base_name_or_path: str = BASE_CHECKPOINT,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        strict: bool = False,
        verify_load: bool = True,
        **icl_overrides,
    ) -> "PI05ICLPolicy":
        """Load frozen pi05_base weights into a fresh pi05_icl policy.

        ``strict=False`` is the default and should stay that way: the ICL
        params (demo encoder, gates) are legitimately missing from the base
        checkpoint. With ``strict=True`` the inherited loader would raise
        inside its broad try/except and silently return an untrained model
        (ICL §4.0.1) — ``verify_load`` re-checks a frozen tensor against the
        checkpoint as cheap insurance either way.
        """
        from .configuration_pi05_icl import icl_config_from_base

        config = icl_config_from_base(
            base_name_or_path, device=device, dtype=dtype, **icl_overrides
        )
        policy = cls.from_pretrained(base_name_or_path, config=config, strict=strict)
        if verify_load:
            assert_weights_loaded(policy, base_name_or_path)
        return policy


# Factory convention (lerobot.policies.factory._get_policy_cls_from_policy_name):
# the policy class is looked up as `<config-class-minus-Config>Policy` in the
# sibling modeling_* module — ICLConfig resolves "ICLPolicy". Alias it.
ICLPolicy = PI05ICLPolicy


def assert_weights_loaded(
    policy: PI05ICLPolicy, base_name_or_path: str = BASE_CHECKPOINT
) -> None:
    """Assert a frozen parameter equals its checkpoint tensor.

    Guards against the swallowed-exception path in
    ``PI05Policy.from_pretrained`` (ICL §4.0.1): a silent failure returns a
    randomly initialized model that still runs.
    """
    from transformers.utils import cached_file

    probe_param = "model.action_in_proj.weight"  # never renamed by the key fixer
    try:
        param = dict(policy.named_parameters())[probe_param]
    except KeyError as e:
        raise AssertionError(f"probe parameter {probe_param} not found on policy") from e

    resolved = cached_file(base_name_or_path, "model.safetensors")
    from safetensors.torch import load_file

    original = load_file(resolved)
    fixed = policy._fix_pytorch_state_dict_keys(original, policy.config)
    fixed = {k if k.startswith("model.") else f"model.{k}": v for k, v in fixed.items()}
    if probe_param not in fixed:
        raise AssertionError(
            f"probe key {probe_param} absent from checkpoint {base_name_or_path}"
        )
    ckpt_slice = fixed[probe_param][0, :8].to(torch.float32).cpu()
    param_slice = param.detach()[0, :8].to(torch.float32).cpu()
    if param.dtype == torch.float32:
        if not torch.equal(ckpt_slice, param_slice):
            raise AssertionError(
                f"weights not loaded: {probe_param} differs from checkpoint"
            )
    else:
        if not torch.allclose(ckpt_slice, param_slice, rtol=0.05, atol=1e-3):
            raise AssertionError(
                f"weights not loaded: {probe_param} differs from checkpoint beyond "
                f"dtype roundoff (param dtype {param.dtype})"
            )


def _merge_lora_state_dict(state_dict: dict, config: ICLConfig) -> dict:
    """Fold LoRA deltas into the base weights, producing PLAIN module names.

    A LoRA-injected policy stores ``...q_proj.base_layer.weight`` plus
    ``...lora_A/B.default.weight``; a freshly constructed policy (what the
    inference server builds via ``from_pretrained``) has none of that. Merged
    weight = W + (alpha/r) * B @ A is mathematically identical at inference,
    so serving checkpoints ship merged, plain-keyed state dicts that
    ``strict=True`` loads cleanly. The adapter-only artifact keeps the
    unmerged LoRA for continued training.
    """
    scaling = config.lora.alpha / config.lora.rank
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if ".lora_A." in key:
            pairs.setdefault(key.split(".lora_A.")[0], {})["A"] = value
        elif ".lora_B." in key:
            pairs.setdefault(key.split(".lora_B.")[0], {})["B"] = value

    merged = {}
    for key, value in state_dict.items():
        if ".lora_" in key:
            continue
        if ".base_layer." in key:
            prefix = key.split(".base_layer.")[0]
            new_key = key.replace(".base_layer.", ".")
            pair = pairs.get(prefix, {})
            if key.endswith(".weight") and "A" in pair and "B" in pair:
                delta = scaling * (pair["B"].float() @ pair["A"].float())
                value = (value.float() + delta).to(value.dtype)
            merged[new_key] = value
        else:
            merged[key] = value
    return merged


def so101_serving_config(
    config: ICLConfig,
    input_feature_rename: dict[str, str],
    state_dim: int = 6,
    action_dim: int = 6,
) -> ICLConfig:
    """Rebuild the policy config under the ROS setup's feature schema.

    ``InferenceEngine.validate_policy_input_features`` requires the loaded
    policy's camera keys to EXACTLY match the client setup's
    (``observation.images.wrist`` etc.) and the state shape to match the
    robot's (6) — the same convention the self-improve lerobot-train
    checkpoints follow via ``--rename_map``. Image shapes stay metadata
    (the model resizes internally); state/action shapes drive unpadding.
    """
    import copy

    from lerobot.configs import FeatureType, PolicyFeature

    new_input = {}
    for key, ft in config.input_features.items():
        new_key = input_feature_rename.get(key, key)
        if key == "observation.state":
            ft = PolicyFeature(type=ft.type, shape=(state_dim,))
        new_input[new_key] = ft
    new_output = {}
    for key, ft in config.output_features.items():
        if key == "action":
            ft = PolicyFeature(type=ft.type, shape=(action_dim,))
        new_output[key] = ft
    cfg = copy.copy(config)
    cfg.input_features = new_input
    cfg.output_features = new_output
    return cfg


def save_serving_checkpoint(
    policy: PI05ICLPolicy,
    out_dir: Path | str,
    *,
    base_name_or_path: str = BASE_CHECKPOINT,
    processor_dir: Path | str | None = None,
    input_feature_rename: dict[str, str] | None = None,
    dataset_stats: dict | None = None,
) -> Path:
    """Write a self-consistent serving checkpoint directory.

    Contains: ``model.safetensors`` (frozen base + LoRA MERGED into the base
    weights + DemoEncoder, plain ``model.``-prefixed keys so the inherited
    remap is a no-op and the server's default ``strict=True`` loads into a
    freshly constructed — non-LoRA-injected — policy; see
    :func:`_merge_lora_state_dict`), ``config.json`` with ``type:
    pi05_icl``, and the processor jsons. ``processor_dir`` overrides
    where ``policy_{pre,post}processor.json`` are copied from; by default
    they come from the base snapshot (DROID stats) — pass ``dataset_stats``
    (the ACTIVE stage's) to build stage-correct processors instead (ICL
    §6.5). ``input_feature_rename`` rebuilds the config under the ROS
    setup's camera keys (see :func:`so101_serving_config`). The base
    checkpoint is never rewritten.
    """
    import shutil

    from transformers.utils import cached_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    merged_state = _merge_lora_state_dict(policy.state_dict(), policy.config)
    save_file(
        {k: v.contiguous() for k, v in merged_state.items()},
        str(out_dir / "model.safetensors"),
    )

    serving_cfg = policy.config
    if input_feature_rename:
        serving_cfg = so101_serving_config(policy.config, input_feature_rename)
    serving_cfg._save_pretrained(out_dir)

    if dataset_stats is not None:
        from .processor_pi05_icl import make_pi05_icl_pre_post_processors

        pre, post = make_pi05_icl_pre_post_processors(serving_cfg, dataset_stats=dataset_stats)
        pre.save_pretrained(out_dir)
        post.save_pretrained(out_dir)
    else:
        src_dir = Path(processor_dir) if processor_dir else Path(cached_file(
            base_name_or_path, "config.json"
        )).parent
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            src = src_dir / name
            if src.exists():
                shutil.copy(src, out_dir / name)
            else:
                logger.warning("processor file %s not found in %s", name, src_dir)

    meta = {
        "policy_type": "pi05_icl",
        "base_checkpoint": base_name_or_path,
        **({"feature_rename": input_feature_rename} if input_feature_rename else {}),
    }
    (out_dir / "icl_config.json").write_text(json.dumps(meta, indent=2))
    return out_dir
