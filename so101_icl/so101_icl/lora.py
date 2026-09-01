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
(``lerobot/policies/pretrained.py:381``), for two reasons:

1. It freezes *all* parameters before wrapping — which would freeze our
   DemoEncoder, whose gates/encoder must train alongside the adapters.
2. It returns a ``PeftModel``, breaking the ``PreTrainedPolicy`` contract
   the inference server relies on.

Instead we freeze the base ourselves and call peft's
``inject_adapter_in_model`` with an explicit list of target module names.

Note that lerobot's own pi05 default targets
(``PI05Policy._get_default_peft_targets``: the action-expert attention plus
the state/action projections and the flow-matching time-MLP/out-projections)
are the *opposite* choice, made for embodiment adaptation. We keep the
action expert and the flow-matching head frozen — adapting them would move
the motor policy away from base pi05 at step 0 and void the M0 zero-init
guarantee (:func:`assert_zero_init`); the ICL adaptation budget goes to the
VLM attention, which must learn to read the injected demo tokens.
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
    EXPERT_ATTENTION_PROJ_PATH,
    EXPERT_ATTN_PROJS,
    VLM_ATTENTION_PROJ_PATH,
    VLM_ATTN_PROJS,
    ICLConfig,
)
from .modeling_pi05_icl import PI05ICLPolicy

logger = logging.getLogger(__name__)

ADAPTER_FILE = "icl_adapter.safetensors"
ADAPTER_META = "icl_adapter_config.json"


def _discover_target_names(core: torch.nn.Module) -> dict[str, list[str]]:
    """Exact LoRA target module names, built from the live module tree.

    Layer indices are enumerated from the actual modules (not a regex over
    names), and every constructed name must exist — a lerobot bump that
    renames or moves the attention projections raises here, at injection
    time, with the expected paths spelled out. Returns
    ``{"vlm": [...], "expert": [...]}``.
    """
    module_names = {name for name, _ in core.named_modules()}
    try:
        layers = core.paligemma_with_expert.paligemma.model.language_model.layers
    except AttributeError as e:
        raise RuntimeError(
            "pi05_icl: the VLM module path changed in this lerobot version "
            f"(expected 'paligemma_with_expert.paligemma.model.language_model."
            f"layers'); update {VLM_ATTENTION_PROJ_PATH!r}."
        ) from e
    names = [
        VLM_ATTENTION_PROJ_PATH.format(i=i, proj=proj)
        for i in range(len(layers))
        for proj in VLM_ATTN_PROJS
    ]
    try:
        expert_layers = core.paligemma_with_expert.gemma_expert.model.layers
    except AttributeError as e:
        raise RuntimeError(
            "pi05_icl: the action-expert module path changed in this lerobot "
            f"version (expected 'paligemma_with_expert.gemma_expert.model."
            f"layers'); update {EXPERT_ATTENTION_PROJ_PATH!r}."
        ) from e
    expert_names = [
        EXPERT_ATTENTION_PROJ_PATH.format(i=i, proj=proj)
        for i in range(len(expert_layers))
        for proj in EXPERT_ATTN_PROJS
    ]

    missing = [n for n in names + expert_names if n not in module_names]
    if missing:
        raise RuntimeError(
            "pi05_icl: constructed LoRA target modules not found in the model "
            f"({len(missing)}/{len(names + expert_names)} missing; first: "
            f"{missing[:3]}). The attention module paths in "
            "configuration_pi05_icl.py no longer match this lerobot version."
        )
    return {"vlm": names, "expert": expert_names}


def discover_lora_targets(core: torch.nn.Module, targets_mode: str) -> list[str]:
    """Target names for the configured ``targets`` mode.

    ``vlm_attention`` adapts only the PaliGemma LM attention q/k/v/o;
    ``vlm_attention+expert`` additionally adapts the action-expert q/v
    (unused by the shipped stage configs — see the module docstring).
    """
    targets = _discover_target_names(core)
    if targets_mode == "vlm_attention":
        return targets["vlm"]
    if targets_mode == "vlm_attention+expert":
        return targets["vlm"] + targets["expert"]
    raise ValueError(f"Unknown lora.targets: {targets_mode!r}")


def lora_target_modules(core: torch.nn.Module, config: ICLConfig) -> list[str]:
    """Backward-compatible entry point used by ``inject_lora``."""
    return discover_lora_targets(core, config.lora.targets)


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
        target_modules=lora_target_modules(policy.model, config),
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
            "LoRA injection matched no modules — the discovered target names "
            "no longer match this lerobot version (see discover_lora_targets)."
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

    The gate logits start at exactly 0 (uniform softmax shares, i.e. every
    effective gate equals ``gate_budget / n_branches``) and the order
    embedding is exactly zero; LoRA ``B`` matrices are zero (peft default).
    The pool/traj out-projections are deliberately NOT zero — see
    ``DemoEncoder.reset_icl_parameters`` (dead-saddle fix).
    """
    enc = policy.model.demo_encoder
    gates = [("gate_vis", enc.gate_vis)]
    if enc.keypoints_enabled:
        gates.append(("gate_kp", enc.gate_kp))
    for label, tensor in gates:
        if tensor.item() != 0.0:
            raise AssertionError(
                f"zero-init violated: demo_encoder.{label} logit is "
                f"{tensor.item()}, expected 0.0"
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
    """Restore adapter tensors into an already-LoRA-injected policy.

    Beyond the base/init fingerprint guards, the recorded LoRA structure
    (rank, alpha, targets mode) and the DemoEncoder config must match the
    current policy config — a mismatch means the adapter tensors cannot be
    interpreted correctly, and it is caught here at load time, not mid-run.
    Dropout is recorded but deliberately NOT enforced: it is regularization
    and may legitimately differ between stages.
    """
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

    recorded_lora = meta.get("lora", {})
    cfg_lora = policy.config.lora
    for field_name in ("rank", "alpha"):
        recorded = recorded_lora.get(field_name)
        current = getattr(cfg_lora, field_name)
        if recorded != current:
            raise ValueError(
                f"adapter {adapter_dir} was trained with lora.{field_name}="
                f"{recorded!r} but the policy config has {current!r}"
            )
    # Targets: equality — or the one widening direction, loading a
    # vlm_attention adapter into a vlm_attention+expert policy (the §9
    # expert-LoRA contingency). Freshly injected expert adapters start at
    # B=0, so the added branch is M0-neutral; the reverse narrowing would
    # silently DROP trained expert adapters and is refused.
    recorded_targets = recorded_lora.get("targets")
    widening = (
        recorded_targets == "vlm_attention"
        and cfg_lora.targets == "vlm_attention+expert"
    )
    if recorded_targets != cfg_lora.targets and not widening:
        raise ValueError(
            f"adapter {adapter_dir} was trained with lora.targets="
            f"{recorded_targets!r} but the policy config has "
            f"{cfg_lora.targets!r} (only vlm_attention -> vlm_attention+expert "
            f"widening is supported)"
        )
    recorded_de = meta.get("demo_encoder")
    current_de = dataclasses.asdict(policy.config.demo_encoder)
    if recorded_de != current_de:
        diff = {
            k: (recorded_de.get(k), current_de.get(k))
            for k in set(recorded_de or {}) | set(current_de)
            if (recorded_de or {}).get(k) != (current_de or {}).get(k)
        }
        raise ValueError(
            f"adapter {adapter_dir} was trained with a different demo_encoder "
            f"config; differing fields (adapter, config): {diff}"
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
    base_name_or_path: str = BASE_CHECKPOINT,
) -> PI05ICLPolicy:
    """Standard training setup — the single init mechanism for both stages.

    Freeze base -> inject LoRA -> initialize: from a prior adapter (stage 2
    finetune / resume; hyperparameters validated on load) or fresh ICL
    parameters plus the structural M0 zero-init assertions (stage 1
    pretrain). The zero-init assertions only make sense from scratch — a
    loaded adapter is by definition no longer at zero — so they are skipped
    when ``init_adapter_from`` is given.
    """
    inject_lora(policy, config)
    if init_adapter_from is not None:
        load_icl_adapter(policy, init_adapter_from, base_name_or_path=base_name_or_path)
    else:
        policy.model.demo_encoder.reset_icl_parameters()
        assert_zero_init(policy)
    return policy
