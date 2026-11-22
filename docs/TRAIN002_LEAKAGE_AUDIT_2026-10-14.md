# TRAIN-002 leakage and interpretation audit

Omer questioned whether improvement at command magnitude 0.60, despite omitting it
from fine-tuning, indicates leakage. Audited 2026-10-14 on `abe218f` plus local report
clarifications. This is a post-hoc audit, not a new confirmatory experiment.

## Assessment

No direct train/test contamination or omitted-branch gradient contribution was found
in the checks below. However, **0.60 was excluded from the production training loss,
not from the entire development process**. Calling it an entirely unseen strength
overstates this experiment. The report now states the narrower claim and the exposure.

The experiment does not beat LeWM's full benchmark or replace its pretraining. It
starts from the released pretrained weights and adapts them to a specialized push
distribution. Targeted training makes ordinary-clip performance slightly worse.

## What was checked

- Compared all ten pairs of five groups: TRAIN-001 train (600 families), validation
  (120 families plus 60 ordinary windows), confirmation (240 families plus 120
  windows), TRAIN-002 ordinary training (2,400 windows), and confirmation (240
  families plus 120 windows). Zero shared archive episodes, challenge reset seeds,
  recipe hashes, history hashes, or full-content hashes between groups.
- Reran `verify_integrity` against each study's saved earlier-study exclusion
  registry and actual file hashes: 1,140 TRAIN-001 and 2,760 TRAIN-002 records passed.
  This also checks within-study duplicate identities. The registry covers earlier
  study/readout families; it is not a reconstruction of upstream pretraining splits.
- The current READOUT-002 source fits on separately sampled rendered poses and selects
  by its own validation error. Study-frame checks are evaluation-only; TRAIN-002
  loads the frozen readouts by hash. This audit inspected that path but did not refit
  the readouts or compare every rendered pose for geometric near-duplicates.
- Inspected all three omitted-arm run records. They use weights
  `(1/3, 1/3, 1/3, 0)`, 600 training families, the original checkpoint's trainable
  parameter digest, the recorded development cache, and final update 2,000. Checkpoint
  hashes match. No best-validation checkpoint is selected by the production runner.
- With the actual model and two actual training families (one per scene), replaced
  every action and target in the 0.60 branch with arbitrary finite values. Under
  three fixed dropout seeds, the loss, retained-branch predictions and all **93
  parameter-gradient tensors were bitwise identical**. Omitted-branch output
  gradients were zero; frozen tensors and BatchNorm statistics stayed unchanged.
  This is a targeted implementation check, not a replay of every training update.
- Inspected rollout and validation paths: true future latents enter the loss only;
  the second prediction receives the first prediction. Validation runs without
  gradients and only logs scores. It does not drive an optimizer or checkpoint choice.
- Verified the sealed sources, inputs and checkpoints and recomputed all estimates
  and intervals from **5,040 saved records**: exact match to both saved summary and
  tracked runtime manifest. Recomputed means at 0.60: frozen **26.8405**, balanced
  **16.2103**, omitted **17.0263**, ordinary-only **27.3057**. Transfer fraction
  **0.9232 [0.8532, 0.9884]**, a descriptive secondary quantity.
- Freshly inferred two confirmation families under the frozen, balanced and omitted
  checkpoints (one training seed): six model/family comparisons. Largest primary
  endpoint-coordinate difference from saved results was **0.00003891 units**. This
  checks a sample of actual checkpoint outputs; it is not a full inference replication.
- Relevant training, collection, provenance and recomputation tests: **59 passed**.

## Why 0.60 is not wholly unseen

1. Training families were admitted using simulator movement at **0.30 and 0.60**
   (`MOVE_REQUIRED`). Thus 0.60 outcomes helped choose which training starts qualified.
2. The learning rate was inherited from the earlier all-strength TRAIN-001 pilot.
   That pilot and its validation included 0.60.
3. The omitted arm's logged validation also includes 0.60. It does not select the
   final checkpoint, but the magnitude is still visible in development diagnostics.
4. The experiment does not remove 0.60 from the released model's original pretraining.

These are restrictions on the support/generalization interpretation. They are not
evidence that fresh confirmation labels were fed into gradient updates. A strict
end-to-end magnitude holdout would need different admission and development rules
and newly reserved confirmation data; this audit does not retrofit such a protocol.

## Why a short run can help, and why the omitted result is plausible

The targeted set is deliberately special: 600 families, half designed starts and half
demo-derived starts, with repeated constant commands and a movement admission rule.
The omitted arm still trains on 0.15, 0.30 and 0.45. It retains everything learned in
expensive upstream pretraining. Improvement on nearby, related pushes is therefore
plausible; the amount of additional compute alone does not diagnose leakage.

An exploratory check of the saved simulator endpoints found:

| Set and scene | Families | Mean separation of 0.45 vs 0.60 endpoints | Fraction within 5.12 units |
| --- | ---: | ---: | ---: |
| Training, designed | 300 | 0.281 | 100% |
| Training, demo-derived | 300 | 14.266 | 66.7% |
| Confirmation, designed | 120 | 0.298 | 100% |
| Confirmation, demo-derived | 120 | 14.666 | 58.3% |

Thus omitting a command value does not necessarily omit its physical outcome regime.
In every designed family checked, 0.45 already produces almost the same block endpoint
as 0.60. This makes transfer less surprising. It does **not** prove what the network
learned, and the demo-derived means show that similarity is not universal.

Remaining limits include related procedural scene distributions, shared readout bias,
unknown upstream pretraining overlap, and 48 versus 64 effective rollouts per update.
Exact identity separation does not rule out all near-duplicates. Original optimization
history is supported by the source and saved provenance, not independently retrained.

## Reproduction and artifacts

Commands run from the repository root (local evidence is ignored by Git):

```sh
UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync python outputs/leakage_audit_20261014/audit.py > outputs/leakage_audit_20261014/result.json
UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync python outputs/leakage_audit_20261014/verify_records.py > outputs/leakage_audit_20261014/records.json
UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync python -m jepa_action_effects.train002_recompute
UV_CACHE_DIR=/private/tmp/jepa-uv-cache uv run --locked --no-sync pytest tests/test_train001_model.py tests/test_train001_collect.py tests/test_train001_provenance.py tests/test_train002.py tests/test_train002_recompute.py
```

Sources: `train002._data`, `train002.train`, `train002.model_registry`,
`train001_model.rollout`, `weighted_loss`, `latent_mse`, `train_run`,
`train001_collect._admit_common`, `verify_integrity`, and DECISIONS D028/D035.
No training, sealed source, checkpoint, data, protocol or confirmatory estimate changed.
