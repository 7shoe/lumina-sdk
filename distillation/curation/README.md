# M100 curation

Additional methods beside DD, DCGM, HMDC, BAS and RAMBO. Their implementations and
past datasets are unchanged. The campaign lives in `distillation/experiments/curation_m100`.

| Acronym | Method | Final targets | Creation models |
|---|---|---|---|
| ITH | Input + target kernel herding within ten demand strata | Original | None |
| PIH | ITH with an equally weighted physical-margin/flow kernel | Original | None |
| KC | Plain input k-center | Original | None |
| DQS | One example from each of 100 demand quantiles | Original | None |
| LDM | Bounded load changes matching conditional input means | Checked OPF solutions | None |
| ALC | Anchored analytical correction using a ridge surrogate | Learned, not guaranteed feasible | Seven T1 teachers, excluding the student architecture |

`methods.py` reuses the exploratory mathematics. `physics.py` adds physical
descriptors without solver calls. `pipeline.py` freezes train-only pools, teacher
predictions and generation progress. Original selections use the existing
selection publisher; `artifacts.py` publishes edited pairs without replacing their
targets. Both are read by `distillation.experiments.datasets.load_dataset`.

Creation uses 4,096 candidate graphs per case and seed 777. This is the final-set
budget M100, not a claim of only 100 labels used during creation. Teacher training,
candidate-label use and solver calls are recorded separately.

ALC's eight variants share the same selected inputs; their teaching targets use
different seven-model committees. They are not 800 distinct input grids.

`demand.py` preserves the pilot's input-matching update. A failed PYPOWER solve is
retried with stationarity tolerance 1e-7 (initially 1e-8); physical feasibility must
still pass 1e-6 and the optimizer must report convergence. Attempts and retries
are recorded. This avoids rejecting converged-to-roundoff points without accepting
unsolved labels or weakening the physical gate. Previous solver code is unchanged.

