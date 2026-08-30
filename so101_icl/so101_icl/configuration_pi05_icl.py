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

"""Configuration for the ``pi05_icl`` policy (demo-conditioned pi0.5).

``ICLConfig`` extends ``PI05Config`` with the ICL additions (demo encoder,
LoRA, gates). It is registered under ``"pi05_icl"`` so the lerobot
convention-based registry resolves it at runtime — no lerobot edits.
"""

from dataclasses import dataclass, field  # noqa: F401 (field used in dataclasses below)

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

BASE_CHECKPOINT = "lerobot/pi05_base"

# Exact module path of the VLM attention inside PI05Pytorch; asserted at
# import in modeling_pi05_icl.py so a lerobot version bump fails loudly.
VLM_ATTENTION_PROJ_PATH = (
    "paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.{proj}_proj"
)

VLM_LORA_TARGETS = (
    r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
)
EXPERT_LORA_TARGETS = r"gemma_expert\.model\.layers\.\d+\.self_attn\.(q|v)_proj"


@dataclass
class KeypointConfig:
    """Keypoint demo representation (Keypoint Action Tokens style, rev 5).

    Demo keyframes are additionally represented as sparse keypoints (SIFT,
    precomputed offline into an npz cache keyed ``<ds_idx>/<episode>``):
    per frame up to ``n_kp`` keypoints × ``kp_dim`` features (2 normalized
    coords + 128 L2-normalized descriptor + 1 valid flag). The DemoEncoder
    pools them into ``tokens_kp`` tokens/demo through their own gated
    branch, giving the prefix correspondence-bearing visual content that
    raw SigLIP pooling cannot express.
    """

    enabled: bool = False
    n_kp: int = 16                 # keypoints per frame (K)
    kp_dim: int = 131              # 2 coords + 128 SIFT descriptor + 1 valid mask
    tokens_kp: int = 32            # pooled keypoint tokens per demo


@dataclass
class DemoEncoderConfig:
    """DemoEncoder architecture knobs (ICL_IMPLEMENTATION.md §7)."""

    k_max: int = 4                 # max support demos per pack
    frames_per_demo: int = 6       # keyframes per demo (F)
    tokens_vis: int = 64           # pooled visual tokens per demo (T)
    tokens_traj: int = 32          # trajectory tokens per demo (2 * traj_steps)
    traj_steps: int = 16           # downsampled state/action steps per demo (S)
    n_heads: int = 8               # attention-pool heads
    hidden_dim: int = 512          # trajectory MLP hidden width
    # One-sided barrier on the gates: effective gate = clamp(gate, min=gate_floor),
    # initialized AT the floor. 0.0 keeps the original zero-init; a positive
    # floor (e.g. 0.1) prevents the optimizer from nulling the whole demo
    # branch through a single scalar (the "null-out" failure mode observed in
    # stage-1 run 2) — the gate can still grow freely.
    gate_floor: float = 0.0
    # Keypoint branch (rev 5); disabled by default so old configs/checkpoints
    # load unchanged. Accepts a plain dict from YAML.
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)

    def __post_init__(self):
        if isinstance(self.keypoints, dict):
            self.keypoints = KeypointConfig(**self.keypoints)


@dataclass
class LoRAConfig:
    """LoRA adapter knobs; injected manually via peft (see lora.py)."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # "vlm_attention" | "vlm_attention+expert"
    targets: str = "vlm_attention"
    # The demo gates start at 0 and need a stronger push than the adapters.
    gate_lr_mult: float = 10.0


@PreTrainedConfig.register_subclass("pi05_icl")
@dataclass
class ICLConfig(PI05Config):
    """PI05Config + ICL additions; ``type`` resolves to ``"pi05_icl"``."""

    demo_encoder: DemoEncoderConfig = field(default_factory=DemoEncoderConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self):
        de = self.demo_encoder
        if de.tokens_traj != 2 * de.traj_steps:
            raise ValueError(
                f"tokens_traj ({de.tokens_traj}) must equal 2 * traj_steps "
                f"({de.traj_steps}): the trajectory branch emits one state and "
                "one action token per step."
            )
        if not 1 <= de.k_max <= 8:
            raise ValueError(f"k_max must be in [1, 8], got {de.k_max}")
        if de.gate_floor < 0:
            raise ValueError(f"gate_floor must be >= 0, got {de.gate_floor}")
        if self.lora.targets not in ("vlm_attention", "vlm_attention+expert"):
            raise ValueError(f"Unknown lora.targets: {self.lora.targets!r}")
        super().__post_init__()


def icl_config_from_base(
    base_name_or_path: str = BASE_CHECKPOINT,
    *,
    device: str | None = None,
    dtype: str = "bfloat16",
    **icl_overrides,
) -> ICLConfig:
    """Build an ``ICLConfig`` carrying the base checkpoint's fields.

    The pi05_base ``config.json`` has ``type: "pi05"``; parsing it directly
    would resolve a plain ``PI05Config``. We parse it as such, then copy every
    field onto ``ICLConfig`` (a strict subclass, so all fields carry over —
    via ``getattr`` so nested ``PolicyFeature``/``RTCConfig`` objects are
    preserved, not dict-round-tripped) and apply the ICL overrides on top.
    """
    import dataclasses

    base_cfg = PI05Config.from_pretrained(base_name_or_path)
    base_fields = {f.name: getattr(base_cfg, f.name) for f in dataclasses.fields(base_cfg)}
    base_fields.update(icl_overrides)
    cfg = ICLConfig(**base_fields)
    cfg.dtype = dtype
    if device is not None:
        cfg.device = device
    cfg.__post_init__()
    return cfg
