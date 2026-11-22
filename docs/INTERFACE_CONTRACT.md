# Checkpoint–simulator interface contract

ENV-001 audit, MODEL-001 inference, and SIM-001 replay validation completed:
**2026-09-26**. Status: **SOURCE-AUDITED CANDIDATE; STRICT CPU INFERENCE,
DETERMINISTIC REPLAY, AND PAIR-SPECIFIC CONTACT GATES PASSED; HISTORICAL
SIMULATOR LINKAGE AND PHYSICAL COMPETENCE UNPROVEN**.

The source/metadata audit is complete, including a real ordinary simulator
transition and checkpoint tensor inspection. This is not proof of the exact
training-time simulator/checkpoint pairing. The published checkpoint has no
training run manifest, simulator commit, dependency lock, or action statistics.
Keep the pair provisional until the open gates below are resolved.

## Immutable artifacts

| Artifact | Pinned revision / evidence |
| --- | --- |
| Model source | [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm/tree/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82), commit `ca231f9f9d9ab041034b6d05e90b6e04bd6cff82` (checkpoint-era reference) |
| Checkpoint | [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht/tree/22b330c28c27ead4bfd1888615af1340e3fe9052), revision `22b330c28c27ead4bfd1888615af1340e3fe9052` |
| Weights | `weights.pt`, 72,290,721 bytes; downloaded SHA-256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`, matching Hub LFS metadata |
| Configuration | `config.json`, SHA-256 `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`; embedded in the runtime manifest |
| Simulator | [`galilai-group/stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel/tree/44c45bd1f774995aac0cc2b41444ebea24c1d5ad), commit `44c45bd1f774995aac0cc2b41444ebea24c1d5ad` (2026-03-23, package `0.0.5`), environment `swm/PushT-v1`, class `stable_worldmodel.envs.pusht.env.PushT` |
| Licenses | [LeWM source MIT](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/LICENSE); [SWM source MIT](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/LICENSE). Checkpoint and dataset cards declare `mit`; neither Hub tree includes a standalone LICENSE. Preserve the relevant copyright/license notices if redistributing; no weights/data are committed here. |
| Accessible training data | [`quentinll/lewm-pusht`](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850), revision `655cd446b9929369d7d406001da85c15d1457850`; verified compressed SHA-256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318` (13,136,247,974 bytes); extracted HDF5 SHA-256 `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb` (46,300,921,856 bytes). |

[Source manifest](manifests/env001-sources.json) records individual source hashes;
[runtime manifest](manifests/env001-runtime.json) records measured versions and
results. Selection of the March simulator is a **provenance-based candidate
choice**, not a claim that its commit was used to collect training data (D005).

## Controls, observations, and timing

Source links below refer to the pinned March simulator and model source.

| Field | Established value / limitation |
| --- | --- |
| Action coordinates and units | Two float32 controls `(x, y)` in declared `[-1, 1]`. World coordinates span 512 units; x points right and y down in the renderer. In relative mode, each control creates a target at `agent.position + 100 * action`. This is a relative target offset tracked by PD control, not a direct velocity assignment. [Controller](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/pusht/env.py#L310), [drawing axes](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/utils.py#L11). |
| Bounds / clipping | The simulator declares the box but does **not** clip actions or target coordinates; its target clipping line is commented out. Callers must enforce candidate bounds before stepping. |
| Physics / control | `dt=0.01` seconds; control/render frequency 10 Hz; 10 physics substeps per control. PD gains `k_p=100`, `k_v=20`; target fixed within each control. Default space gravity `(0,0)` and damping `0`. [Initialization](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/pusht/env.py#L29). |
| Native model step | Five consecutive 2-D controls flattened in temporal order to width 10; not necessarily one action repeated five times. Frames sampled every five controls, therefore one native prediction step = 0.5 seconds. [Training data config](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/config/train/data/pusht.yaml), [dataset indexing](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/data/dataset.py#L160). The study's provisional two-step horizon is 10 controls / 1 second, still subject to the simulator-only pilot rule. |
| Action normalization | All 2,336,736 archive rows are finite. Training statistics are mean `[-0.0078125652, 0.0068606865]`, sample SD `[0.2084674835, 0.2067486644]`; evaluation statistics are mean `[-0.0078125644, 0.0068606872]`, population SD `[0.2084674428, 0.2067486264]`. MODEL-001 uses the evaluation convention selected before inference (D006). Normalize each control before grouping five controls. [Training normalizer](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/utils.py#L15), [evaluation](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/eval.py#L69). |
| Image contract | RGB uint8 HWC, 224×224 render produced from a 512×512 pygame surface via OpenCV resize; no crop. Keep default colors, T shape, `with_target=True`, `render_action=False`. Eval transforms: CHW, float32 /255, ImageNet channel mean `(0.485,0.456,0.406)` and SD `(0.229,0.224,0.225)`, then resize to 224. Local implementation uses torchvision bilinear resize with antialiasing; already-224 images need no resampling. Training uses stable-pretraining's `ToImage` / `Resize`, whose exact historical version is unrecorded. [Render](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/pusht/env.py#L403), [eval transforms](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/eval.py#L17). |
| History | Three frames, separated by five controls: for current time `t`, use `t-10, t-5, t`. Two executed action blocks connect them; the next candidate block leaves `t`. No padding/repeated-frame history assumed. [Training config](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/config/train/lewm.yaml). |
| Proprioception | Simulator observation `proprio = [agent_x, agent_y, agent_vx, agent_vy]`; `state = [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]` in the pinned source and EVAL-001 integration check. The audited JEPA encode/predict path consumes **pixels and action only**. Dataset availability of proprio/state does not make them model inputs. Future proprioception is unnecessary and must not be supplied. [Encoder](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/jepa.py#L29). |
| Latent | Tiny ViT with patch size 14 and 192-wide CLS embedding, followed by projector; predictor and prediction projector also output width 192. Real readout input should be `encode(...)["emb"]` after the projector, aligned to predicted embeddings, not patch tokens or unprojected CLS. Weights show `(1,257,192)` image-position embeddings and `(1,3,192)` predictor-position embeddings. No fitted readout yet. |
| Termination / alignment | Base step returns post-control observation, `terminated` from agent+block positional distance <20 and angular difference <π/9, and `truncated=False`. Wrappers may add limits/autoreset; audit uses the base class and rejects any early end. The collector left-shifts stored actions to align `o_t` with the action leaving it. [Collector alignment](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/world.py#L513). |

## Model API caveat and leakage-free handoff

The Hub config names newer symbols `stable_worldmodel.wm.lewm.LeWM` and
`module.Predictor`. The pinned March runtime exports `wm.lewm.JEPA` and
`ARPredictor`. **Direct Hydra instantiation of the Hub config is not supported by
this runtime.** MODEL-001 explicitly maps those names to the pinned March `JEPA`,
`ARPredictor`, `Embedder`, and `MLP` classes and constructs the Hugging Face tiny
ViT directly. The resulting model has exactly 303 state entries and loads strictly
with no missing or unexpected keys.

The [March rollout](https://github.com/lucas-maes/le-wm/blob/ca231f9f9d9ab041034b6d05e90b6e04bd6cff82/jepa.py#L61)
accepts pixels `(B,S,H,3,224,224)` and action sequence `(B,S,T,10)`, where
`H=3` and `T` **includes H−1 historical blocks**. For K future predictions,
provide T=H−1+K blocks: the two executed historical blocks followed by the K
candidate blocks. It returns `(B,S,H+K,192)`, including encoded context.
The loop count `T-H` plus one final prediction explains this otherwise ambiguous
API. For K=2, T=4; the future predictions are indices 3 and 4. The current upstream
rollout instead separates `action_history` from future candidates; do not mix APIs.

MODEL-001 exposes only observed pixels, executed past actions, and candidate
future actions to the predictor. Future real frames belong exclusively to target
encoding/evaluation. It recomputes action embeddings per candidate, uses evaluation
mode with frozen parameters and inference mode, and matches an independent manual
autoregressive implementation exactly for one- and two-step predictions.

## Simulator state and contacts for SIM-001

Reset calls seeded variation sampling and rebuilds the Pymunk space. It temporarily
sets a goal state, renders a goal image, then restores the start. Each `_set_state`
calls `space.step(0.01)`, so reset already includes physics evolution. Replay the
whole reset and identical warm-up; do not replace it with a pose assignment.

The seven-component state is `[agent_x, agent_y, block_x, block_y, block_angle_mod_2pi,
agent_vx, agent_vy]`. The local variable `vel_block` in `_set_state` actually sets
**agent** velocity. The state omits block velocity/angular velocity, forces,
collision solver caches, and other dynamic state. Read body position, angle,
velocity, angular velocity, force/torque and contact history for replay checks;
pose-only restoration is insufficient. [Reset and state setter](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/pusht/env.py#L235).

All shapes use default collision type 0 and one `(0,0)` `post_solve` callback.
`n_contacts` includes walls, aggregates contact points over a control, and the March
`_get_info` additionally has an incorrect divisor (`1 / dt * control_hz = 1000`,
versus 10 actual substeps). **Never use it as an agent–block contact label.**
SIM-001 now inspects arbiter bodies against `env.agent` and `env.block` after every
physics substep while preserving the existing callback through subclass dispatch.
It records reset, warm-up, and candidate phases separately, stable shape identities,
contact geometry, and complete observable body dynamics. Agent is a kinematic circle;
the dynamic T-block has two polygon shapes, and both are covered by the validated
pair-specific label. [Collision setup](https://github.com/galilai-group/stable-worldmodel/blob/44c45bd1f774995aac0cc2b41444ebea24c1d5ad/stable_worldmodel/envs/pusht/env.py#L582).

`run_family` recreates the seed/options and full warm-up independently for every
candidate. It validates raw controls without clipping, returns history at `t-10`,
`t-5`, and `t`, supports five- or ten-control futures, and stops/marks a candidate
incomplete on termination. Future frames remain separate from observed history.
The recorded state covers position, unwrapped angle, linear/angular velocity,
force/torque, and sleep status; Pymunk solver caches remain unexposed and are not
claimed as cloned state.

Compared with September source `63988116d34cde56aea1240d5e58eb158ac67dc0`, the
March source has older observation bounds, contact divisor, symmetry logic, and
shape-scale handling. Both have effective default T scale 30: March hardcodes it
inside `add_tee`; September sets the variation default to 30. This comparison
corrects any inference that the two default T geometries necessarily differ.

## Measured checks and remaining gates

Python 3.10.21 / macOS 15.7.3 arm64; PyTorch 2.14.0 CPU, torchvision 0.29.0,
Pymunk 7.3.0, pygame 2.6.1, Gymnasium 1.3.0, OpenCV headless 4.11.0.86,
Shapely 2.1.2. Full dependency resolution is in `uv.lock`. These are measured local
audit versions, **not recovered training-time requirements**. No MPS/CUDA run,
model instantiation, accelerator comparison, or throughput profile was performed.
Upstream training config requests GPU/bfloat16; that is not evidence inference
requires a GPU.

- Downloaded checkpoint: 303 state-dict entries, all finite; five interface-critical
  tensor shapes checked. Safe `torch.load(..., weights_only=True)` inspection only.
- Real seed-42 trajectory: 15 controls; no termination/truncation; moving agent,
  stationary block. History at controls 0/5/10 has shape `(1,3,3,224,224)`; target at
  15 is held separately with shape `(1,1,3,224,224)`. Two raw past blocks `(2,10)`
  and one raw future block `(1,10)`; deliberately **not normalized model inputs**.
- Three behavioral tests passed: physics timing/relative action semantics,
  history-target separation/action ordering, RGB channel scaling and type guard.
- SIM-001 ran seeds 0–15 with six fixed candidates under duplicate, reverse, and a
  recorded random order. All candidates completed. Contact identities/timestamps,
  flags, frames, and observable dynamics matched; maximum numerical and positional
  replay discrepancy were both exactly zero on this runtime. Instrumented and base
  simulator trajectories also matched exactly.
- Direct-push, sustained-contact, near-miss, block-wall-only, and moving/rotating
  fixtures passed. The sustained fixture contacted both T polygons; the wall-only
  fixture had 100 wall-contact substeps and zero agent–block-contact substeps.
  Annotated diagnostics remain separate from model-compatible RGB frames.
- `make check` passed. Initial OpenCV 4.14 wheel emitted duplicate SDL class
  warnings with pygame; pinning 4.11.0.86 removed those warnings. Gymnasium's
  upstream array-casting warning remains and did not fail the checks.

ENV-001, MODEL-001, and SIM-001 close the interface, frozen-inference, and replay /
contact-label gates, not scientific compatibility. Before the main study:

1. PILOT-001: combine replay families, EVAL-001 metrics, and frozen model inference;
   use simulator outcomes rather than model errors to select contact/near-miss cases.
2. PILOT/FREEZE: audit training overlap using accessible records. No independence
   claim is established by new seeds alone; no study/test families were collected.

MODEL-001 found 18,685 episodes and no embedded simulator lineage. Float32 CPU
inference passed manual-rollout equality, future-target isolation, candidate-order,
frozen-parameter, BatchNorm, and finite-output checks. The seed-42 two-step latent
MSE was 0.00853 versus 0.01864 for latent persistence; 32 preselected training-
archive windows gave 0.00497 versus 0.35487. These are ordinary-transition and
training-data diagnostics, not evidence of physical-effect prediction or
generalization. See `docs/manifests/model001-runtime.json`.
SIM-001 measurements and artifact hashes are in
`docs/manifests/sim001-runtime.json`; detailed traces and annotated examples remain
ignored under `outputs/sim001/`.

Reproduce the local audit after downloading the two checkpoint files at the pinned
revision into `checkpoints/lewm-pusht/`:

```sh
make setup
make check
make test
UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked python -m jepa_action_effects.interface_audit
UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --locked python -m jepa_action_effects.simulator_audit
```

The audit verifies file hashes before loading tensors. Generated report and image
are under `outputs/env001/`; the compact measured report is tracked in the manifest.
Weights, source archives, images, and future training-data downloads stay ignored.

## 2026-09-28 note: reset with a wall-overlapping `goal_state`

Measured during NEXT-001 implementation, simulator only. `reset` applies
`_set_state(goal_state)` and then `_set_state(state)`, and neither call re-indexes
collision shapes.

When the goal pose overlaps a wall, as `FAR_GOAL` (block at 450, 450) does, the
second physics step moves the requested start block. In one measured case the block
moved from (300, 250, 0.4 rad) to (299.64, 251.71, 0.40007) with zero residual
velocity. Using the start state as its own goal reproduces the requested pose
exactly.

Earlier families are unaffected, because every label comes from realized physics
states. Designed challenge clearances were therefore approximate to about 2 units.
DIST-001 replays use the kick-free reset. The drawn green goal overlay comes from the
variation space (default 256, 256, π/4), not from `goal_state`.
