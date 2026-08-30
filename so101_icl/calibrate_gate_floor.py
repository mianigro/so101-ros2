"""One-off: calibrate demo gate_floor from real prefix token norms (rev 5).

Loads pi05_icl on cuda, pushes ONE real batch from the stage-1 registry
through embed_prefix twice (bare, and with a real demo pack), and reports
the RMS of the three prefix blocks — camera tokens, language tokens, and
demo tokens at the current floor. Demo tokens leave LayerNorm at RMS≈1,
so their RMS ≈ the effective gate: the recommended floor for a target
"demo loudness" fraction f of the camera tokens is simply f * camera_RMS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import yaml

from so101_icl.configuration_pi05_icl import DemoEncoderConfig, LoRAConfig
from so101_icl.data import ICLDataset, load_task_registry, open_local_dataset
from so101_icl.modeling_pi05_icl import PI05ICLPolicy
from so101_icl.train_loop import _move_icl_fields, _split_icl_fields, build_stage_preprocessor

STAGE = "so101_icl/configs/icl_pretrain_droid_v1.yaml"
REGISTRY = "so101_icl/configs/task_registry_droid.json"

stage = yaml.safe_load(Path(STAGE).read_text())
policy = PI05ICLPolicy.from_base(
    stage["base_checkpoint"], device="cuda", dtype="bfloat16",
    demo_encoder=DemoEncoderConfig(**stage.get("demo_encoder", {})),
    lora=LoRAConfig(**stage.get("lora", {})),
)
policy.eval()

dataset = ICLDataset(REGISTRY, policy.config, split="train",
                     demo_camera=stage["dataset"]["demo_camera"],
                     keypoint_cache=stage["dataset"].get("keypoint_cache"))
reg = load_task_registry(REGISTRY)
primary = open_local_dataset(reg["datasets"][0]["repo_id"], reg["datasets"][0]["root"])
preprocessor, _ = build_stage_preprocessor(policy.config, primary.meta.stats)

item = dataset[0]
raw = {k: (v[None] if torch.is_tensor(v) else [v])
       for k, v in item.items() if not k.startswith("icl.")}
batch_bare = preprocessor(raw)
icl_fields = {k: v[None] for k, v in item.items()
              if k.startswith("icl.") and torch.is_tensor(v)}
batch_demo = {**batch_bare, **_move_icl_fields(icl_fields, "cuda")}
batch_bare = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch_bare.items()}

captured = {}
orig = policy.model.embed_prefix
def recorder(images, img_masks, tokens, masks):
    out = orig(images, img_masks, tokens, masks)
    captured["embs"] = out[0].detach().float()
    captured["n_lang"] = masks.shape[1]
    return out
policy.model.embed_prefix = recorder

def rms(t):
    return float(t.pow(2).mean().sqrt())

def blocks():
    embs, n_lang = captured["embs"][0], captured["n_lang"]
    n_img = embs.shape[0] - n_lang
    return embs[:n_img], embs[n_img:n_img + 0], embs[n_img - 0:], n_img, n_lang  # placeholder

with torch.no_grad():
    policy.forward(batch_bare)                       # no pack: [cam | lang]
    embs = captured["embs"][0]; n_lang = captured["n_lang"]; n_img = embs.shape[0] - n_lang
    cam_rms, lang_rms = rms(embs[:n_img]), rms(embs[n_img:])

    n_img_bare, cam_tok_bare = n_img, embs[:n_img].shape[0]
    batch_demo = {k: v.to("cuda") if torch.is_tensor(v) else v
                  for k, v in batch_demo.items()}
    policy.forward(batch_demo)                       # with pack: [cam | demo | lang]
    embs = captured["embs"][0]
    n_demo = embs.shape[0] - cam_tok_bare - n_lang
    demo_rms = rms(embs[cam_tok_bare:cam_tok_bare + n_demo])

print(f"token counts: camera={cam_tok_bare}, demo={n_demo}, language={n_lang}")
print(f"RMS  camera={cam_rms:.4f}  language={lang_rms:.4f}  demo(gate=floor)={demo_rms:.4f}")
print(f"demo/camera loudness ratio: {demo_rms / cam_rms:.3f}")
floor_now = stage["demo_encoder"]["gate_floor"]
for frac in (0.2, 0.3, 0.5):
    print(f"floor for demo@{frac:.0%} of camera tokens: {frac * cam_rms:.4f}  (current {floor_now})")
