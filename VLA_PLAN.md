# VLA Integration Plan: in-context learning on the robot

**Goal:** the model deployed on the robot consumes Bridge Robot's task
context — subtask DAG + few-shot demonstration packs — the way GEN-1.5
(Generalist AI, closed) does: learn a new task from seconds of
demonstration in context, no gradient updates at deployment.

**Starting point:** we run **π0.5** (open weights via
[openpi](https://github.com/Physical-Intelligence/openpi)). π0.5 has **no
in-context demonstration mechanism**: its inputs are camera images + a
language instruction, and openpi's adaptation path is fine-tuning
(PI guidance: ~1–20 hours of demo data per environment/task). Prompt-only
conditioning is the only thing it supports out of the box.

**What this repo already provides** (both paths build on it, no upstream
changes needed):

- **Dispatch package** — `GET /api/mission/<ws>/<id>/dispatch-package`
  (`ros2_action_goal:v1`): per-subtask action specs with dependency edges,
  each carrying a **conditioning pack**:
  `{episode_id, segment_id, t_start, t_end, caption, annotation,
  outcome, embodiment, objects[], keyframes[], video_asset_ref,
  trajectory{available, asset_ref}, prov_id}`.
- **LeRobot dataset export** — per-observation episodes with ffmpeg-cut
  clips, trajectory parquet sliced to the time window and mapped to
  `observation.state` / `action` features, plus a provenance manifest.
  Trajectories optional end-to-end.
- **5-second segment observations** with `t_start`/`t_end` provenance —
  inside GEN-1.5's 3–12 s demonstration sweet spot.

Reference points: GEN-1.5 (one-shot, 3–12 s demo in a 30 s context window,
optional 1–10 gradient steps); [RICL](https://arxiv.org/html/2508.02062v1)
(post-hoc in-context adaptation for π0/OpenVLA — 10–20 demos, zero
gradient updates, the published open recipe this plan follows);
[Retrieval-VLA](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhang_Retrieval-VLA_Training-Free_In-Context_Adaptation_for_Vision-Language-Action_Models_CVPRF_2026_paper.pdf)
(CVPR 2026, frozen VLA + demo buffer); [ICRT](https://icrt.dev/)
(natively in-context open alternative).

---

## Path 1 — Prompt + keyframe conditioning (zero training, bridge work only)

**Thesis:** π0.5 already consumes language and images. Feed it the pack's
language (the annotation that worked, objects, outcome) as an enriched
instruction, and the pack's keyframes as additional image observations.
This is task *disambiguation and grounding*, not imitation — the model
never sees the demonstrated motion. Use it to validate the bridge and the
retrieval loop end-to-end, and to establish the baseline that Path 2 must
beat.

### Bridge mapping (pack → π0.5 inputs)

| Dispatch-pack field | π0.5 input |
|---|---|
| `subtask.skill_spec` + `annotation` + `objects[]` | Language instruction, e.g. `"<skill_spec>. As demonstrated: <annotation>. Objects involved: cup, saucer. (demo outcome: success)"` |
| `keyframes[]` (overlay variant preferred) | Spare image-observation slots, ordered by `t` (start → goal). ContextVLA-style multi-frame conditioning. |
| `outcome: failure` packs | Either dropped by the bridge or passed as negative language ("avoid the failure mode in the demo: …") — decide empirically. |
| `depends_on` edges | Sequential subtask scheduling in the bridge; only the active subtask's pack is in context. |
| `video_asset_ref`, `trajectory` | **Unused in Path 1.** π0.5 cannot consume video or trajectories in context. |

### Scope of work

ROS 2 bridge node only (the future consumer this repo's README already
names). No model changes, no training, no new endpoints upstream. The
bridge: subscribes/pulls the dispatch package, resolves keyframe asset
refs (presigned URLs), maintains the active-subtask state machine, and
issues π0.5 action-chunk queries per its standard openpi inference loop.

### Expected behavior and honest limits

- Better task specification than a bare prompt: correct objects named,
  demonstrated scene layout visible in keyframes, known-good phrasing.
- **Will not teach new skills.** If π0.5 cannot roughly do the motion,
  keyframes will not fix it. Failure mode to watch: the model imitating
  the *static* keyframe scene rather than the action.

### Validation gates (exit criteria for Path 1)

1. End-to-end run on real hardware: objective → plan → approval → packs →
   bridge → π0.5 executes a task it already knows (e.g. a pick-and-place
   variant), comparing bare-prompt vs pack-conditioned success rate over
   ≥ 10 trials each.
2. Measurable lift (or documented parity) from pack conditioning on
   instruction-following details (correct object, correct destination).
3. Latency budget: pack resolution + keyframe fetch ≤ ~1 s per subtask
   switch (all local S3).

Path 1 is a milestone, not the destination. Its outputs feed Path 2:
every trial is a candidate new episode (self-demo capture), and its
baseline numbers are the comparison point.

---

## Path 2 — RICL-style post-hoc in-context modification of π0.5

**Thesis:** replicate GEN-1.5's behavior openly by following the published
RICL recipe on π0.5: **freeze the base entirely**, add a demonstration
encoder + adapter that injects demo representations into the VLM context,
train **only the adapter** on demonstration→action pairs. At deployment:
frozen π0.5 + adapter adapts to new tasks from ~10–20 retrieved demos with
zero gradient updates.

### Architecture

```
                        ┌────────────────────────── frozen π0.5 ──────────┐
 demo keyframes ───────►│ demo encoder (new)                              │
 demo trajectory ──────►│ (vision backbone + lightweight temporal pool)    │
 (when present)         │        │ demo prefix tokens                       │
                        │        ▼                                         │
 live cameras ─────────►│ PaliGemma VLM ──► flow-matching action expert ──►│ actions
 instruction ──────────►│   ▲ (prefix/adapter injection)                   │
                        └───┬──────────────────────────────────────────────┘
                            └── adapter (new, trained) ── knowledge-insulated
```

- **Frozen:** π0.5 VLM backbone and action expert, exactly as released.
- **Demo encoder (new):** encode one demonstration pack into a small set
  of prefix tokens. Input: ordered keyframes (+ `t_start/t_end` ordering),
  optionally the trajectory slice when available (1-D temporal profile or
  downsampled state sequence). Reuse π0.5's own vision tower for the
  frames (frozen) so the encoder itself is light.
- **Adapter + injection (new):** LoRA-scale adapters on the VLM attention
  that attend from the model's normal token stream to the demo prefix —
  the knowledge-insulation pattern π0.5 already uses between its VLM and
  action expert (gradients flow through adapters only).
- **Context budget:** demo prefix targeted at a few hundred tokens per
  demo, top-k packs per subtask (k ≤ 4) — the 30-second-context regime
  GEN-1.5 demonstrated, sized to π0.5's practical image-token budget.
- **Trajectory-optional preserved:** the encoder takes video-derived
  features always, trajectory features when `trajectory.available` — a
  masking input, mirroring how the whole stack treats trajectories.

### Training data — produced by this repo, no new pipeline

- **Source:** the Phase 7 dataset builder. Curate sets of demonstrations
  per task/scene with trajectories, freeze, export → LeRobot episodes
  (clip + `observation.state`/`action` per 5 s window, provenance).
- **Adapter objective (RICL formulation):** given *support* demos for a
  task and a *query* observation from the same task, predict the query's
  actions. Sample support/query splits across our exported episodes;
  hold out entire tasks (not just episodes) for the in-context
  generalization test.
- **Volume to start:** RICL reports adaptation from 10–20 demos per task;
  the adapter pre-training itself uses whatever we can export — begin
  with existing teleop sessions plus any public LeRobot-format data we
  can filter through the dataset builder (outcome/object criteria give
  the quality gate).
- **Open question to resolve early:** openpi's training stack is JAX.
  Either implement the adapter in openpi's training loop (cleanest for
  weight compatibility) or export to PyTorch for adapter training and
  re-fuse for serving. Decide in the first spike; the decision gate is
  round-tripping a modified forward pass against the released weights.

### Hardware

- Adapter-only training: 3B-class frozen base + small adapters fits the
  local RTX 4080 16 GB (frozen weights can be partially off-loaded;
  LoRA-scale trainable parameters).
- Robot-side inference: unchanged footprint from π0.5 plus the demo
  encoder (light); prefix tokens add modest context cost.

### Runtime integration — the dispatch package is the demo buffer

Per subtask, the bridge already receives `conditioning_pack.demonstrations`
ranked by the retrieval specialist. In Path 2 the bridge:

1. selects top-k demonstrations (success outcomes first),
2. fetches keyframes/trajectories (local S3, presigned),
3. builds the demo prefix for the adapter,
4. runs frozen-π0.5-with-adapter in the normal control loop.

No upstream changes: the same `ros2_action_goal:v1` contract serves
Path 1 and Path 2; the bridge toggles the conditioning mode.

### Milestones

1. **Spike (1–2 weeks):** forward-pass injection proof — get any demo
   prefix attended-to by frozen π0.5 in openpi (JAX or PyTorch port
   decision here). Gate: modified forward pass runs and reproduces the
   unmodified model's outputs with zeroed adapters.
2. **Data readiness (parallel):** first frozen LeRobot dataset from real
   teleop through the Datasets tab; adapter sampling code over it.
3. **Adapter v1:** train on our exports; evaluate held-out *tasks*
   in-context vs (a) Path 1 baseline and (b) π0.5 bare prompt. Gate:
   in-context beats bare-prompt on unseen tasks; approach Path 1 on seen
   tasks.
4. **On-robot loop:** bridge serves packs → adapter conditions π0.5;
   capture outcomes; successful runs ingested as new episodes (self-demo
   flywheel — the designed future phase).
5. **Optional polish:** GEN-1.5-style "1–10 gradient steps" fast mode on
   top (adapter-only fine-tuning from in-context initialization) — cheap
   to try once v1 exists.

### Risks

- **JAX/PyTorch round-trip friction** (openpi training stack) — bounded
  by the spike's decision gate.
- **Prefix token budget vs π0.5's image-token limits** — mitigate by
  keyframe-count ablation (k, frames per demo) in adapter v1 training.
- **Video-only demos under-train the trajectory branch** — the masking
  design keeps it optional; if under-used, drop trajectory features
  entirely (consistent with the stack's trajectories-optional stance).
- **RICL's recipe was demonstrated on π0/OpenVLA, not π0.5** — π0.5's
  co-training/knowledge-insulation should make injection *easier* (same
  principle), but this is the plan's main technical unknown; the spike
  and milestone 3 gate it explicitly.

---

## Sequencing and decision points

1. **Now:** Path 1 through the bridge (validates the whole loop, zero
   model risk, produces baseline numbers + candidate episodes).
2. **Parallel:** dataset builder produces the first adapter-training
   export from real teleop.
3. **Next:** Path 2 spike → adapter v1 → on-robot in-context evaluation
   vs Path 1 baseline.
4. **Standing alternative:** if adapter progress stalls, swap the bridge's
   policy to a natively in-context open model (ICRT-class) — the
   `ros2_action_goal:v1` contract and LeRobot exports stay valid as-is.
