# ROBUST-001 result — is the dose-dependent compression a readout artifact?

**Status:** complete, exploratory (D034). All records were already opened, and no model
was run. Numbers come from `docs/manifests/robust001-runtime.json`; the generated tables
are in `ROBUST-001_RESULT_DRAFT.md`.

## Question

Every physical number so far decodes predicted latents with readouts fitted on real
frames. Could the frozen model's dose-dependent compression of block displacement come
from one readout's extrapolation instead of from the predictor?

## Design

- **Five instruments.**
  - The three READOUT-002 MLPs (these can saturate).
  - The ridge readout (linear extrapolation).
  - A new kNN readout: the mean block position of the 8 nearest of 50k real training
    frames. It clamps to the bank. k was chosen on validation before any study latent
    was decoded.
- **Data.** SHIFT-002 (development 120, reserved 240 families) and PHYS-002 (development
  300, reserved 150).
- **Off-manifold flag.** A predicted step-2 latent is flagged if its distance to the
  nearest real bank latent exceeds the 95th percentile for encoded real futures of the
  same group.
- **Gate.** All 1,888 shared CALIB-001 values were reproduced first (maximum difference
  8.9e-16).

## Results

**The dose trend appears in all five instruments.** Each row is the reserved SHIFT-002
energy-ratio contrast (high − low dose), with family-bootstrap 95% CIs:

| Scene | MLP primary | MLP 04 | MLP 06 | Ridge | kNN |
| --- | --- | --- | --- | --- | --- |
| Challenge, 0.60 − 0.30 | −0.64 [−0.89, −0.41] | −0.71 [−0.97, −0.47] | −0.62 [−0.90, −0.39] | −0.58 [−1.09, −0.16] | −0.59 [−0.88, −0.33] |
| Archive, ×2 − ×1 | −0.35 [−0.48, −0.23] | −0.47 [−0.67, −0.30] | −0.35 [−0.48, −0.21] | −0.54 [−0.88, −0.30] | −0.58 [−0.89, −0.36] |

- The D034 rule for an instrument-robust trend is met. It is also met without kNN.
- The projection slope b_rev falls with dose for all five instruments in challenge
  scenes. In archive scenes it falls for four; for ridge it is −0.04 [−0.14, 0.05].
- Alignment changes are small and mixed (−0.06 to +0.10). The trend is in magnitude.
- Energy-ratio *levels* depend on the instrument. Readout noise inflates Σ|p|², so ridge
  reads 1.6–2.2. Only trends and inner-product slopes are comparable across instruments.

**The PHYS-002 projection slope does not depend on the instrument.** On the reserved
challenge families, b_rev is:
- 0.561 [0.519, 0.606] for the primary MLP;
- 0.549–0.569 for the other two MLPs;
- 0.543 [0.490, 0.598] for ridge;
- 0.574 [0.524, 0.626] for kNN.

The ordinary cohort stays at 0.82–0.88.

**High-dose predictions are not more off-manifold.**
- Flag rates are 4–12%, close to the 5% expected for real latents by construction.
- They *fall* with dose: from 0.12 to 0.04 (challenge, 0.30 → 0.60) and from 0.07 to
  0.04 (archive, ×1 → ×2).
- On-manifold stratum: four of five instruments keep a CI-excluded negative trend in both
  scenes. For ridge in challenge scenes it widens to −0.36 [−0.88, 0.09]. Under the D034
  rule the stratum verdict is therefore **inconclusive**.

**Instrument disagreement.** On predicted and real latents the disagreement is similar in
archive scenes. In challenge scenes it is larger on predicted latents at high dose:
median spread 23.7 against 16.1 units at 0.60.

## Interpretation

- Readout extrapolation is an unlikely sole explanation. Linear, saturating and clamping
  readouts agree in sign and rough size, and the high-dose predictions sit no further
  from real latents than low-dose ones.
- This is **corroboration, not validation** (D031). All five instruments use the same
  frozen encoder and the same 50k random-reset training frames, so they are not
  independent measurements. Agreement does not prove that any of them reads predicted
  latents correctly.
- The trend is magnitude compression with roughly constant alignment, consistent with
  CALIB-001 (corrected). No mechanism follows.

## Limitations

- Exploratory: every family was already opened. The frozen model only; TRAIN-001
  predicted latents were not saved.
- The on-manifold stratum is selected on model output.
- A readout-free independent instrument (MEAS-001) remains unbuilt.

## Reproduction

```bash
.venv/bin/python -m jepa_action_effects.robust001 all
```

It needs the ignored `outputs/{readout002,shift002,phys002,calib001}`. Stage times on
Omer's Mac CPU: knn 1.6 s, shift 13.6 s, phys 16.2 s, report 0.3 s.
