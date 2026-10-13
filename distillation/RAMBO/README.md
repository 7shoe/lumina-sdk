## Baseline: "Scalable Bilevel Optimization for Generating Maximally Representative OPF Datasets"

Authors: Ignasi Ventura Nadal, Samuel Chevalier
Year: 2024

For joint experiments with DD/DCGM/HMDC, use the shared
[YAML and launch guide](../experiments/configs/README.md). RAMBO generates once per
case/load-domain/seed; fresh or pretrained students use the common evaluation.
Verified physical labels are preserved in reusable `dataset.pt` exports on Lustre.
The standalone generator and its original artifacts remain available.

#### Reproducibility
We reimplement RAMBO for OPFData graphs, fixed dataset splits and resumable runs. The bilevel structure, relaxed KKT conditions and final AC-OPF solve follow the paper, with explicit adaptations from the [reference code](https://github.com/SamChevalier/RAMBO/tree/10952cf812bbd378c3e54fc84abf82806a02660a).

- **Different data integration:** A training-only OPFData template supplies the graph format; nominal loads come from PGLib. Generated examples are training data; validation/test remain unchanged.

- **Narrower load range:** Defaults are 0.8–1.2 times nominal demand; the paper uses 0.5–2.0 and the GitHub example 0.5–1.5. This does not reproduce OPFData’s sampling distribution.

- **Numerical changes:** We derive KKT stationarity symbolically and add a small log floor. An extra OPF warm start improves initialization; `warm_start: feasible` uses the paper’s initialization instead. Local solutions can differ.

- **Verified history:** Following the paper, only checked final OPF solutions enter the reference library; the released code uses relaxed points. We also reject near-duplicates.

- **Bounded, resumable runs:** We add checkpoints and finite retry/time budgets. Exhausted runs remain explicitly incomplete.

- **Different evaluation:** Lumina models train after generation; they never guide RAMBO. Coverage reporting separates individual active constraints from distinct joint patterns.

#### Reference

The paper at 
```bash
@inproceedings{nadal2023scalable,
  title={Scalable bilevel optimization for generating maximally representative OPF datasets},
  author={Nadal, Ignasi Ventura and Chevalier, Samuel},
  booktitle={2023 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

The codebase at
```bash
https://github.com/SamChevalier/RAMBO
```
