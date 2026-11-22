# TRAIN-001 — Can magnitude-balanced fine-tuning correct prediction errors?

## Summary

Run a bounded intervention study using the existing LeWM checkpoint. Compare the frozen model with two fine-tuned copies that receive **identical simulator transitions but different command-magnitude weights**.

The research question is: **Does emphasizing larger commands improve physical predictions beyond ordinary magnitude weighting, without degrading ordinary-case performance?**

Complete a local pilot followed by fresh confirmation, within **24 cumulative machine-hours**. Improvement is not required for successful completion; an interpretable negative result is valid. This replaces the separate-MLP suggestion in REPORT-001.

## 1. Experimental design and data

**Three conditions**

| Condition | Training |
|---|---|
| Frozen baseline | Original checkpoint |
| Archive-weighted fine-tuning | Weight command magnitudes according to their frequency in a separate archive reference sample |
| Magnitude-balanced fine-tuning | Give each of four command magnitudes equal weight |

Both trained conditions use identical initial weights, data, batches, optimizer settings and update counts. Call the first training condition **archive-weighted**, not a reproduction of original archive training.

**Shared branch bank**

- Collect **600 training families**, **120 validation families**, and **240 reserved confirmation families**. Each split contains equal numbers of designed challenge and archive-derived initial states.
- Each family has four constant-control branches at magnitudes **0.15, 0.30, 0.45 and 0.60**. Keep three observed history frames and the existing two-step horizon: ten future controls, or one second.
- Challenge directions follow the existing approach geometry. Archive directions use the normalized mean of the recorded ten-control future; reject near-zero means.
- Reuse validated reset/warm-up replay, bounds enforcement and archive identity checks. Accept families only when all branches complete and the 0.30 and 0.60 branches each move the block more than 5.12 units. Selection uses simulator outcomes only.
- Collect an additional **60 validation and 120 confirmation ordinary archive windows**, without filtering for block movement, to measure regression on ordinary cases.
- Derive archive weights from **1,000 separate archive episodes**, using mean control norm over ten-control windows and bins separated at **0.225, 0.375 and 0.525**. Normalize bin counts into four weights; require every bin to contain at least 100 valid windows.

Keep source episodes, reset families and related branches disjoint across all partitions, including the archive weight sample. Exclude previously used study families and reserved collections. Record recipe/content hashes in addition to seed IDs. Archive-derived observations may overlap the original checkpoint’s pretraining distribution; do not describe them as unseen during pretraining.

**Primary comparison:** balanced versus archive-weighted fine-tuning on the high-magnitude branches, 0.45 and 0.60. Compare against the frozen model as a required accompanying result.

## 2. Training implementation and pilot

Implement reusable training, collection and analysis code under `src/jepa_action_effects/`, with a staged CLI:

`audit → collect-development → cache → pilot → train → collect-confirmation → seal → confirm → report`

Preserve completed experiment modules and their sealed artifacts.

**Model boundary**

- Freeze the visual encoder, visual projector, action normalization and all existing readouts.
- Fine-tune the predictor, action encoder and prediction projector—approximately **11.7 million parameters**.
- Keep BatchNorm running statistics frozen. Enable gradients for the selected parameters and normal predictor dropout during training.
- Cache real projected embeddings once. Store observed history and future targets separately.
- Implement differentiable autoregressive prediction: the second prediction consumes the first predicted embedding, never the real future embedding. Future targets enter only the loss.
- Before training, verify that cached-embedding rollout matches the existing pixel-based frozen inference path.

**Optimization defaults**

- Three paired training seeds: **20261111, 20261112 and 20261113**.
- AdamW, weight decay **0.01**, gradient-norm clipping **1.0**, float32.
- Batch: **16 families × four magnitudes**, with eight families from each scene category.
- Loss: latent MSE averaged over both future steps and latent coordinates, then magnitude-weighted within each family. Families and scene categories receive equal weight.
- Full training: **2,000 optimizer updates per run**, six runs total. Use final checkpoints for the primary analysis; retain intermediate checkpoints for diagnostics only.
- Pair batch order and random seeds across training conditions.

**Pilot and feasibility gate**

Use 60 training and 60 validation families from their eventual development partitions.

1. Verify gradients, frozen parameters, finite outputs and rollout equivalence.
2. Demonstrate optimization on a 16-family overfit diagnostic: at least 50% training-loss reduction within 500 updates.
3. Compare learning rates **1e-5, 3e-5 and 1e-4**, using 250 updates per condition and the first training seed. Select one shared learning rate by the lowest equally weighted validation latent MSE averaged across both conditions; break ties in favor of the lower rate.
4. Measure collection, encoding, training and evaluation throughput. Proceed only when projected remaining runtime, with **50% contingency**, fits the remaining 24-hour budget.
5. Check development readout validity and data coverage before reserving the final protocol.

A synthetic CPU microbenchmark already measured approximately 0.10 seconds per warmed-up update for 64 sequences. Treat this only as feasibility evidence; the real-data pilot determines the runtime estimate.

Start on CPU. If necessary, use an available GPU after numerical-parity checks and record its environment separately. If neither option fits the budget, stop with a pilot report; do not silently reduce seeds or confirmation counts.

## 3. Evaluation and interpretation

**Primary metric**

For each family, average terminal block-position error over the two high-magnitude branches. Compare conditions on the same families, average across the three training seeds, and give the two scene categories equal weight.

Report:

- Balanced minus archive-weighted error.
- Balanced minus frozen error.
- Archive-weighted minus frozen error.
- Separate results for every training seed and scene category.

Use **2,000 paired, scene-stratified family-bootstrap draws**. These intervals describe family uncertainty conditional on the three trained runs; report training-seed variation separately.

**Secondary measurements**

- Absolute endpoint and displacement errors, with distinct labels.
- Projection slope, predicted/real slope ratio, displacement length and direction errors by magnitude.
- Real-future readout controls and latent prediction MSE.
- Correct versus action-permuted predictions.
- Ordinary-window endpoint errors to detect regression.
- Sensitivity to the other two existing READOUT-002 MLPs, without selecting whichever readout favors an intervention.

Include persistence and a cheap calibration comparator: fit one nonnegative scalar on training families to rescale the frozen model’s decoded displacement. Fit it by family-weighted least squares, with no test labels. Label it as a state-supervised output correction, not a trained latent predictor.

**Measurement and claim gates**

- Require at least **50 moved families per primary scene × magnitude cell**.
- Require each cell’s real-frame projection-slope interval to lie within **(0.85, 1.15)**.
- Require real-frame displacement error to remain below **25% of the median meaningful true displacement** in each primary cell.
- Failed measurement gates make physical conclusions inconclusive. Retain all numerical results and latent diagnostics; do not relax gates after confirmation.
- Claim a balanced-training benefit only when both balanced-minus-archive-weighted and balanced-minus-frozen primary intervals lie below zero.
- Predeclare **1.28 workspace units** as the ordinary-case regression tolerance. Report whether the upper interval bound for balanced-minus-frozen endpoint error stays below it. This is a new tolerance for this study, not a retroactive change to previous gates.
- If readout sensitivity changes the conclusion, report the result as measurement-dependent. Agreement among readouts does not fully establish validity on predicted latents.
- A slope closer to one alone does not establish better physical prediction. No planning-benefit claim follows from this study.

Proceed to confirmation when implementation, measurement and budget gates pass, even if the pilot shows no improvement.

## 4. Provenance, tests and deliverables

**Confirmation protection**

Seal the protocol, partitions, source hashes, archive weights, selected learning rate, all six checkpoints, readouts and cache fingerprints before confirmation inference. Use strict provenance validation on both initial execution and resume.

Extend validation to verify the entire model registry and expected family coverage. Reject missing, duplicate, foreign or stale records. A changed checkpoint, input, configuration or source requires a new run identity. Do not overwrite completed evidence.

**Required tests**

- Only the intended prediction parameters change; frozen encoder/projector weights and BatchNorm statistics remain identical.
- Changing future targets cannot change forward predictions.
- Two-step action alignment and gradient flow through the first prediction are correct.
- Weighted losses match analytic fixtures; both arms receive identical batches and updates.
- Episode/family overlap and content duplicates are rejected.
- Zero-length vectors have undefined direction; sparse cells fail gates.
- Bootstrap resamples families jointly across conditions and keeps training seeds distinct.
- Mutation tests reject altered checkpoints, probe bytes, caches and confirmation records.
- Save/resume preserves training state and sampling order on the same device.

Run `make check` and relevant tests after implementation, plus the full suite before sealing because this adds scientific training and evaluation logic.

**Deliverables**

- Training CLI, automated tests and a reproducible runbook.
- Pilot feasibility report and frozen experiment configuration.
- Six fine-tuned checkpoints and caches under ignored output directories.
- Tracked compact manifests, learning curves, magnitude-response figures, paired error comparisons and a result report.
- Updated task board, logbook and scientific decision record.
- Revised REPORT-001 narrative incorporating the actual training results within the existing five-page report and 15-minute presentation limits.

## Assumptions and boundaries

This is a single-checkpoint, single-horizon intervention study. It excludes encoder training, new architectures, new training objectives, additional checkpoints and closed-loop planning. Both training arms use the same transitions; only magnitude weighting changes.

The 24-hour cap covers cumulative collection, encoding, training and evaluation, including the pilot. It excludes implementation time. A budget or feasibility failure produces a documented pilot outcome rather than an unannounced protocol change.

The study supplies a substantial PyTorch training component for the course, subject to instructor acceptance. It can strengthen a potential paper, but publication novelty and generality remain separate questions.
