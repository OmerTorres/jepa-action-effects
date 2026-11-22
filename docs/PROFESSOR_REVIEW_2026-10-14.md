# Final course-report review — 2026-10-14

## Assessment and scope

**Very close to submission; make two scientific wording corrections and one caption
correction first. Provisional written-package assessment: approximately 92/100.**
The rewritten report is substantially easier to understand: concrete examples, questions
that motivate the experiments, explicit training work, and a clear adaptation result.
The scientific scope remains a strong course project. No additional GPU experiment,
training run or new cohort is needed to resolve this review.

Reviewed local `6aec9fc4ade97cee4dcf31e5960637a07126a2b2`, whose tree equals merged
`origin/main` at `87f1e9e`, including changes since the previously accepted `10770ff`.
PR #6 is merged. The original working tree was clean. This review changes only review
notes/task memory, not the report, code, manifests, models or sealed evidence.

The estimate is not an official grade. The handout supplies criteria, not numeric
weights. The presentation and individual mastery remain ungraded. Contributions will
be discussed verbally under the professor's explicit instruction; their absence from
the written report is not a finding. The handout is a source of grading requirements,
not authorization to submit the work.

## Required wording corrections

### F1 — P2: scope the unchanged-effect claim to the evidence

`reports/report001/main.tex:310–313` infers that the same physical effect is predicted
as smaller across both scene types because median displacement barely changes. Similar
unqualified plateau language appears at lines 214–217 and 453–454.

The near-equal medians do not mean each family's physical effect stays fixed. A read-only
check of all 300 saved SUPPORT-001 families, after verifying each NPZ against its
collection hash, gives the following for strength 0.60 versus 1.00:

| Scene | Families | Mean true displacement at 0.60 | At 1.00 | Families whose true endpoints differ by more than 5.12 | Largest endpoint difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Designed | 150 | 24.0879 | 24.0008 | 0 | 1.6294 |
| Demo | 150 | 84.7732 | 96.4449 | 34 | 179.7783 |

These are descriptive audit calculations, not a new confirmatory analysis. They do not
invalidate the reported gain trend. They do invalidate a blanket assertion that the
physical effect is unchanged in demo scenes.

Suggested replacement: "Median true displacement changes little in either scene type.
In designed scenes, the true endpoints also remain nearly unchanged, yet the predicted
gain falls as commands grow. Some demo families have substantially different true
outcomes at stronger commands, so their stable median does not imply identical effects."
Qualify the setup's plateau explanation as applying to the illustrated designed pushes,
and carry that scope through the discussion. Keep the sealed numerical estimates intact.

### F2 — P2: distinguish corroboration from ruling out the instrument

`reports/report001/main.tex:148–151` says the error is not in the instrument or command
reading. The E4 heading at line 331 and its Table 1 answer at line 184 similarly exclude
the instrument categorically.

The report itself correctly acknowledges at lines 263–264 and 336–340 that the probes
share an encoder/frame bank, imagined-latent-specific bias can remain, and the
readout-free instrument did not pass its engineering checks. D034 explicitly calls
agreement corroboration rather than validation on predicted latents. A near-one
aggregate agent gain also does not identify or exclude a mechanism in the action pathway.

Suggested headline: "The trend is robust across readouts and differs between agent and
block." Suggested E4 heading: "The trend survives alternative readouts." Table answer:
"All five readouts agree; shared bias remains possible." Retain the existing caveats
and E5's descriptive status. No additional probe or experiment is requested.

### F3 — P3: correct the validation-set label

`reports/report001/main.tex:416` calls Figure 4(c)'s cohort "120 designed validation
families." The curves use all 120 TRAIN-001 validation families: **60 designed and 60
demo**. This is explicit in `docs/manifests/train001-runtime.json` at
`collection.development.counts` and in the provenance note of
`docs/manifests/training-curves.json`.

Change the caption to "120 validation families (60 designed, 60 demo)." The comment at
`src/jepa_action_effects/reporting.py:1537` repeats the same misdescription; correct that
comment as well. The plotted values are correct.

## Checks passed and practical limits

- `make check && make test`: Ruff, formatting and doctor passed; **362 tests passed**,
  25 warnings, 18.38 seconds for pytest. Warnings concern Gymnasium input casting and
  joblib core-count detection.
- `git diff --check 10770ff..HEAD`: passed. `git fetch origin` followed by
  `git diff --stat HEAD origin/main`: empty; local report matches merged main.
- Inspected the handout and all five pages of `outputs/report001/pdf/main.pdf` using
  Poppler renders. Five pages including references; no clipping or overlaps found.
- Created a clean tracked-source export using `git archive HEAD` in
  `/private/tmp/jepa-final-review-20261014/clean`. Ran `jepa_action_effects.reporting`
  there using the existing locked Python 3.10.21 environment and exported `src` on
  `PYTHONPATH`: **111 numbers and four figures** rebuilt without checkpoint/archive
  artifacts. `numbers.tex`, the claims ledger and `inputs.json` matched byte-for-byte.
- Compiled that export with Tectonic and the existing TeX cache. The rebuilt PDF has
  five pages; extracted text and all five 1800-pixel Poppler page renders match the
  submitted local PDF exactly. Font-request/underfull-box warnings were non-blocking.
  This validates Route A using a clean source tree, not a fresh-machine environment
  install or an independent rerun of the experiments.
- Independently checked D039 outcome-only selection over all 150 SUPPORT designed and
  120 TRAIN designed families: selected examples match `s1-challenge-077` and
  `challenge-conf-059`. No selection by model error was found. Projection bars and
  seed averaging are consistent with the stated definitions.
- Primary-source checks of all 11 bibliography entries found no actionable citation
  or attribution error. Relevant sources include [LeWM](https://arxiv.org/abs/2603.19312),
  [Intervention Gap](https://arxiv.org/abs/2608.29998),
  [AD-WM](https://arxiv.org/html/2609.30264v2),
  [Do-JEPA](https://arxiv.org/abs/2609.37378),
  [PhysWeep](https://arxiv.org/html/2609.06207), and
  [Kang et al.](https://arxiv.org/abs/2310.00873). This checks attribution, not independent
  reproduction of those papers' findings.
- TRAIN-001 training-data/model reuse is disclosed; TRAIN-002 confirmation is fresh.
  No new finding on that reuse. The E2 structure contrast also had a prespecified
  equivalence band, so its near-zero result is not merely a nonsignificance argument.
- The substantial PyTorch training, comparisons, reproducibility instructions and
  report sections satisfy the course requirements in this reviewer's assessment.
  One checkpoint/environment, the short horizon, shared probes, the omission-arm
  batch-size confound, and the unestablished planning penalty still limit the claims.

Optional, non-blocking documentation cleanup: `REPRODUCE.md:144` says every input hash
is verified during example export. `report_examples.py:166` reads the SUPPORT prediction
NPZ and later records its digest, but does not compare that digest before use. The
current file matches the tracked digest, and the required export-diff check detects
changes; no incorrect current value was found. Narrow the wording or add an explicit
comparison in a future provenance cleanup.

## Reproducing F1 without model inference

Run from the repository root with the shared ignored SUPPORT artifacts:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from jepa_action_effects.pilot_collection import sha256_file

root = Path('outputs/support001')
collection = json.loads((root / 'collection.json').read_text())
assert collection['doses'] == [0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
for scene in ('challenge', 'archive'):
    differences, displacements = [], []
    for entry in collection['families'].values():
        if entry['scene'] != scene:
            continue
        path = root / entry['relative_path']
        assert sha256_file(path) == entry['file_sha256']
        with np.load(path) as arrays:
            start = arrays['history_states'][2, 2:4]
            ends = arrays['future_states'][:, -1, 2:4]
            differences.append(float(np.linalg.norm(ends[5] - ends[3])))
            displacements.append(np.linalg.norm(ends - start, axis=1))
    differences = np.asarray(differences)
    means = np.asarray(displacements).mean(axis=0)
    print(scene, len(differences), means[[3, 5]],
          int((differences > 5.12).sum()), float(differences.max()))
PY
```

## Handoff

The actionable findings are also posted in the
[PR #6 review comment](https://github.com/OmerTorres/jepa-action-effects/pull/6#issuecomment-6160331296)
for the implementing agent. These local review notes have not been committed.

Implement F1–F3 as report wording/caption corrections, regenerate and inspect the
five-page PDF, and submit the exported report after the authors' final read. Preserve
all sealed evidence. Further research belongs to the separate paper roadmap; the
current course submission does not need additional GPU work.
