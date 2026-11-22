# Test plan

`make check` runs lint, formatting, and the numerical/rendering environment smoke
check. `make test` currently includes ENV-001 behavioral checks for physics timing,
relative actions, history/target isolation, action ordering, and RGB preprocessing.
These checks do not establish replay equivalence, contact validity, or model quality.

Add behavioral tests alongside the first implementations:

- Reset/warm-up replay, branch-order independence, and collision-pair identity.
- Oracle zero error/regret; persistence, doubled/reversed effects, spurious motion.
- No-contact moving blocks, zero vectors, termination, clipping, and time alignment.
- Family-disjoint splits, family-level aggregation, and deterministic bootstrap seeds.
- Predictor inputs exclude future observations and future proprioception.
