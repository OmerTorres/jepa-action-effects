# PREDICTOR-GALLERY-001 — Reserved block-pose gallery comparison

Executed 2026-09-28. **The prespecified central positive-evidence rule did not pass.**
On 40 reserved challenge families, the frozen predictor favored the realized
block-position template more than predictions for other actions, but its
advantage over latent persistence was inconclusive and its mean signed gallery
margin was negative. This is a measured, checkpoint-specific dissociation
between global action correspondence and the stricter block-position criterion;
it is not a claim that the model lacks block information.

## Protocol and integrity

The development run reused the 30 GALLERY-001 tuning families (20 challenge,
10 ordinary). The final implementation and development analysis were sealed
before opening the 60 READOUT-001 reserved families (40 challenge, 20 ordinary).
Confirmation was run regardless of the development result. Every family had
six candidates and a two-native-step endpoint. The 10.24-unit gallery was
primary; 20.48 units was secondary. Each eligible endpoint used four
geometry-valid negative translations, a true template, and two separate
nonidentical queries. The agent, goal, background, and block orientation were
held fixed within each gallery. The latent numerical floor was `1e-12`.

All 360 reserved endpoints replayed and restored their saved terminal RGB.
The primary nonidentical-query gate passed in both cohorts: challenge
240/240 eligible, 100% top-1 and fifth-percentile margin ratio 0.854;
ordinary 116/120 eligible, 98.75% top-1 and ratio 0.833. Each cohort had at
least ten families with four eligible candidates. The secondary gate also
passed (240/240 challenge, 115/120 ordinary; both 100% top-1). Duplicate and
reversed-candidate inference on 12 preselected families in each phase, and
duplicate/reversed gallery encoding on the 12 reserved verification families,
had maximum absolute discrepancy zero. The four preselected overlays per phase
were visually inspected. Full per-candidate distances, rejected geometry
attempts, labels, cache fingerprints, and artifact hashes are in the ignored
`outputs/predictor_gallery001/` tree.

## Predictor results

Signed margin is nearest-wrong-template distance minus true-template distance
in projected CLS mean-squared units. A positive margin selects the realized
block position. Each wrong-action control scores the other five whole-sequence
predictions against the *same* endpoint gallery; their average is the exact
uniform-derangement expectation. Estimates average candidates within families
and families within cohorts. Intervals are 2,000 family bootstrap percentile
intervals.

| Reserved cohort and scale | Families / eligible endpoints | Matched top-1 | Matched margin | Matched − persistence margin | Matched − wrong-action margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Challenge, 10.24 (primary) | 40 / 240 | 53.75% [48.32, 59.58] | −0.00923 [−0.01927, −0.00005] | 0.00131 [−0.00807, 0.00959] | **0.04272 [0.03287, 0.05159]** |
| Ordinary, 10.24 | 20 / 116 | 88.33% [80.81, 95.00] | 0.02306 [0.00865, 0.03947] | 0.00979 [−0.00077, 0.02501] | 0.01639 [0.00962, 0.02461] |
| Challenge, 20.48 (secondary) | 40 / 240 | 69.17% [62.50, 75.83] | 0.04557 [0.02328, 0.06705] | 0.01710 [−0.00465, 0.03810] | 0.08886 [0.06836, 0.10912] |
| Ordinary, 20.48 | 20 / 115 | 95.83% [90.83, 100] | 0.11732 [0.08326, 0.16060] | 0.01377 [−0.00694, 0.04596] | 0.02964 [0.01104, 0.05680] |

At the primary challenge scale, persistence top-1 was 48.75% and wrong-action
top-1 was 28.00%; uniform five-way guessing is 20%. The matched model's
53.75% top-1 and positive wrong-action contrast show action-sensitive gallery
correspondence. Its negative mean margin and unresolved persistence contrast
fail two of the four central conditions. The wider gallery improves absolute
retrieval but does not resolve the persistence contrast and cannot substitute
for the primary scale. Ordinary accuracy is high partly because block motion
is limited; its persistence contrast is likewise inconclusive.

On the same reserved families, the LATENT-001 global projected-CLS comparison
remained positive: challenge correspondence advantage `A=0.6111`
`[0.5738,0.6498]` and persistence improvement `B=0.7886`
`[0.7429,0.8339]`; ordinary `A=0.1648 [0.0896,0.2571]` and
`B=0.0781 [0.0261,0.1463]`. These are family-level bootstrap intervals.
Mean true agent displacement was 159.60 workspace units, so global latent
correspondence can reflect agent motion. Per-family contact, motion, and
meaningful-effect strata appear in `confirmation.json`. Challenge strata each
had 40 contributing families. Ordinary `both` contact and meaningful-effect
strata had only four and three families, respectively, and are unassessable
under the ten-family rule. Participating endpoints were deduplicated within
family and regime.

## Study synthesis

| Claim | Evidence | Status |
| --- | --- | --- |
| The simulator branches and frozen inference path can be replayed consistently. | [PILOT-001 logbook evidence](LOGBOOK.md), this run's 360 exact confirmation endpoint replays and zero discrepancy verification | Supported for this simulator/checkpoint pair |
| Global projected-CLS predictions correspond to candidate actions and outperform latent persistence. | [LATENT-001](LATENT-001_RESULT.md), replicated `A` and `B` on the 60 reserved families above | Supported for global embeddings |
| Real-frame latents support the required continuous physical position readout. | [READOUT-001 logbook evidence](LOGBOOK.md), [DELTA-001](DELTA-001_RESULT.md) | Failed gates; physical competence unassessable |
| Controlled rendered block-position galleries are distinguishable by the encoder. | [GALLERY-001](GALLERY-001_RESULT.md), independently passed confirmation query gate | Supported at the two fixed gallery scales |
| The predictor uses the matching action better than mismatched actions on the primary challenge gallery. | Reserved margin contrast `0.04272 [0.03287, 0.05159]` | Supported, conditional on the realized gallery context |
| The predictor beats persistence and reliably selects the realized block position at the primary scale. | Persistence interval crosses zero; mean matched margin is negative | Central positive-evidence rule failed |
| The gallery method itself is new, or block errors have measured action-selection regret. | [Related-work review](GALLERY-001_RELATED_WORK.md); physical readout and regret remain unresolved | Unsupported |

## Claim boundary and next decision

The [GALLERY-001 related-work review](GALLERY-001_RELATED_WORK.md) identifies
prior rendered feature-space pose retrieval and action-discrimination methods.
The new observation here is limited to this frozen checkpoint and these
families: global action correspondence and an advantage over wrong actions
coexist with failure of the stricter primary block-pose rule. The fixed rendered
agent reduces visible nuisance variation, but the nonlinear global embedding
does not isolate agent and block information. Galleries condition on realized
future nuisance state and contain synthetic poses; their scores are not
continuous position errors or deployable action choices. The failed real-frame
readout gates from PILOT-001, READOUT-001, and DELTA-001 still make physical
competence unassessable. Historical training-simulator identity is unproven.

Keep main-study expansion paused. A future study would need a new, separately
scoped measurement or model question and fresh reserved evidence; these 60
families are now opened and cannot serve that role again.

## Execution

The final sealed experimental run used Python 3.10.21, the locked environment,
CPU float32 inference, and float64 reductions. Its cumulative ledger recorded
112.273 seconds: 13.139 development, 99.048 confirmation, 0.003 reporting,
plus audit overhead; the cap was 1,740 seconds for computation plus 60 for
reporting. Final artifacts occupy 25.11 MB: 12.29 MB confirmation renders,
3.46 MB confirmation gallery embeddings, 0.55 MB encoding records, 4.33 MB
confirmation family scores, 2.26 MB development records/examples, and 1.58 MB
new real/prediction cache, with the remainder in manifests and examples. An
earlier development-only run
under a prior code hash was preserved in the ignored
`outputs/predictor_gallery001-preliminary/` directory, then the final run
started from an empty output root before confirmation was opened. The tracked
runtime manifest records hashes and verification commands.
