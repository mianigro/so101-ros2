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

"""Manual LoRA injection for pi05_icl (ICL §4.3).

We deliberately do NOT use lerobot's ``wrap_with_peft``
(``lerobot/policies/pretrained.py:381``): it freezes *all* parameters before
wrapping — which would freeze our DemoEncoder — and returns a ``PeftModel``
that breaks the ``PreTrainedPolicy`` contract the inference server relies
on. Instead we freeze the base ourselves and call peft's
``inject_adapter_in_model`` with an explicit target regex (the default pi05
targets reference modules that do not exist in pi05, e.g. ``state_proj``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model

from .configuration_pi05_icl import (
    BASE_CHECKPOINT,
    EXPERT_LORA_TARGETS,
    ICLConfig,
    VLM_LORA_TARGETS,
)
from .modeling_pi05_icl import PI05ICLPolicy

logger = logging.getLogger(__name__)

ADAPTER_FILE = "icl_adapter.safetensors"
ADAPTER_META = "icl_adapter_config.json"


def lora_target_modules(config: ICLConfig) -> str:
    if config.lora.targets == "vlm_attention":
        return VLM_LORA_TARGETS
    if config.lora.targets == "vlm_attention+expert":
        return rf"({VLM_LORA_TARGETS}|{EXPERT_LORA_TARGETS})"
    raise ValueError(f"Unknown lora.targets: {config.lora.targets!r}")


def inject_lora(policy: PI05ICLPolicy, config: ICLConfig) -> PI05ICLPolicy:
    """Freeze the base, then inject LoRA adapters in place.

    The outer ``PI05ICLPolicy`` type is preserved for the inference server;
    adapters land inside ``policy.model`` at the VLM attention projections.
    """
    for name, param in policy.model.named_parameters():
        if not name.startswith("demo_encoder"):
            param.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=lora_target_modules(config),
    )
    inject_adapter_in_model(lora_config, policy.model)

    # peft's tuner marks ONLY adapter params trainable during injection
    # (peft/tuners/tuners_utils.py:1057 `_mark_only_adapters_as_trainable`),
    # which silently froze the DemoEncoder + gates — re-enable them.
    for name, param in policy.model.named_parameters():
        if name.startswith("demo_encoder"):
            param.requires_grad_(True)

    n_adapters = sum(1 for n, _ in policy.model.named_parameters() if ".lora_" in n)
    if n_adapters == 0:
        raise RuntimeError(
            "LoRA injection matched no modules — the target regex no longer "
            "matches this lerobot version (see VLM_LORA_TARGETS)."
        )
    logger.info("injected %d LoRA parameter tensors into policy.model", n_adapters)
    return policy


def trainable_parameters(policy: PI05ICLPolicy):
    """Yields (name, param) with requires_grad — LoRA + DemoEncoder + gates."""
    for name, param in policy.named_parameters():
        if param.requires_grad:
            yield name, param


def assert_zero_init(policy: PI05ICLPolicy) -> None:
    """Structural M0 assertions: every demo-token pathway starts neutral.

    The gates start AT ``demo_encoder.gate_floor`` (0.0 by default; a
    positive floor is the null-out mitigation, ICL §2.1) and the order
    embedding is exactly zero; LoRA ``B`` matrices are zero (peft default).
    The pool/traj out-projections are deliberately NOT zero — see
    ``DemoEncoder.reset_icl_parameters`` (dead-saddle fix).
    """
    enc = policy.model.demo_encoder
    floor = enc.config.gate_floor
    for label, tensor in [
        ("gate_vis", enc.gate_vis),
        ("gate_traj", enc.gate_traj),
    ]:
        # tolerance: the floor is a python float, the gate is fp32
        if abs(tensor.item() - floor) > 1e-6:
            raise AssertionError(
                f"zero-init violated: demo_encoder.{label} is {tensor.item()}, "
                f"expected gate_floor={floor}"
            )
    if enc.order_emb.weight.abs().max().item() != 0.0:
        raise AssertionError("zero-init violated: demo_encoder.order_emb.weight")
    for name, param in policy.model.named_parameters():
        if name.endswith("lora_B.default.weight") and param.abs().max().item() != 0.0:
            raise AssertionError(f"zero-init violated: LoRA B not zero at {name}")


# ---------------------------------------------------------------------- #
# Adapter-only checkpoints (stage artifacts, <100 MB)                    #
# ---------------------------------------------------------------------- #


def _file_fingerprint(path: Path) -> str:
    """Fast content fingerprint (size + sha256 of first/last MiB)."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if path.stat().st_size > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return h.hexdigest()


def _base_fingerprint(base_name_or_path: str) -> str:
    from transformers.utils import cached_file

    resolved = Path(cached_file(base_name_or_path, "model.safetensors"))
    return _file_fingerprint(resolved)


def save_icl_adapter(
    policy: PI05ICLPolicy,
    out_dir: Path | str,
    *,
    base_name_or_path: str = BASE_CHECKPOINT,
    init_adapter_path: Path | str | None = None,
) -> Path:
    """Write adapter-only artifact: trainable tensors + metadata.

    Stage 2 initializes from the stage-1 artifact via :func:`load_icl_adapter`;
    the metadata records the base checkpoint fingerprint (and, for stage 2,
    the init-adapter fingerprint) so mismatched pairs are refused.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: param.detach().cpu().contiguous()
        for name, param in trainable_parameters(policy)
    }
    save_file(tensors, str(out_dir / ADAPTER_FILE))
    meta = {
        "policy_type": "pi05_icl",
        "base_checkpoint": base_name_or_path,
        "base_fingerprint": _base_fingerprint(base_name_or_path),
        "lora": {
            "rank": policy.config.lora.rank,
            "alpha": policy.config.lora.alpha,
            "dropout": policy.config.lora.dropout,
            "targets": policy.config.lora.targets,
        },
        "demo_encoder": dataclasses.asdict(policy.config.demo_encoder),
    }
    if init_adapter_path is not None:
        meta["init_adapter_fingerprint"] = _file_fingerprint(Path(init_adapter_path))
    (out_dir / ADAPTER_META).write_text(json.dumps(meta, indent=2))
    logger.info("saved %d adapter tensors to %s", len(tensors), out_dir)
    return out_dir


def load_icl_adapter(
    policy: PI05ICLPolicy,
    adapter_dir: Path | str,
    *,
    base_name_or_path: str = BASE_CHECKPOINT,
    expected_init_adapter: Path | str | None = None,
) -> PI05ICLPolicy:
    """Restore adapter tensors into an already-LoRA-injected policy."""
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / ADAPTER_META).read_text())
    if meta.get("base_checkpoint") != base_name_or_path and meta.get(
        "base_fingerprint"
    ) != _base_fingerprint(base_name_or_path):
        raise ValueError(
            f"adapter {adapter_dir} was trained against a different base checkpoint"
        )
    if expected_init_adapter is not None and meta.get("init_adapter_fingerprint") not in (
        None,
        _file_fingerprint(Path(expected_init_adapter)),
    ):
        raise ValueError(
            f"adapter {adapter_dir} was not initialized from {expected_init_adapter}"
        )

    state = load_file(str(adapter_dir / ADAPTER_FILE))
    model_keys = dict(policy.named_parameters()).keys()
    unknown = [k for k in state if k not in model_keys]
    if unknown:
        raise ValueError(
            f"adapter has {len(unknown)} keys not present in the model (config "
            f"mismatch?); first: {unknown[:3]}"
        )
    missing = [
        k for k, _ in trainable_parameters(policy) if k not in state
    ]
    if missing:
        raise ValueError(f"adapter file is missing {len(missing)} trainable keys; first: {missing[:3]}")
    policy.load_state_dict(state, strict=False)
    logger.info("loaded %d adapter tensors from %s", len(state), adapter_dir)
    return policy


def setup_trainable_policy(
    policy: PI05ICLPolicy,
    config: ICLConfig,
    *,
    init_adapter_from: Path | str | None = None,
) -> PI05ICLPolicy:
    """Standard training setup: freeze base -> inject LoRA -> load init adapter."""
    inject_lora(policy, config)
    if init_adapter_from is not None:
        load_icl_adapter(policy, init_adapter_from)
    else:
        policy.model.demo_encoder.reset_icl_parameters()
    assert_zero_init(policy)
    return policy
