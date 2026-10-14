## Baseline: "Scalable Bilevel Optimization for Generating Maximally Representative OPF Datasets"

Authors: Ignasi Ventura Nadal, Samuel Chevalier
Year: 2024

For joint experiments with DD/DCGM/HMDC, use the shared
[YAML and launch guide](../experiments/configs/README.md). RAMBO generates once per
case/load-domain/sampler/seed/count; fresh or pretrained students use the common evaluation.
Verified physical labels are preserved in reusable `dataset.pt` exports on Lustre.
The standalone CPU generator supports PGLib cases **14, 30, 57, 118 and 500**.
It also exports `dataset.pt`, `artifact.json` and `recipe.json` immediately on
completion, without training any student. Load these with
`distillation.synthetic.artifacts.load_dataset(run_directory)`, as for DD/DCGM/HMDC.

Prepare separate **M=100 and M=800** libraries (plus a matched uniform-load control):

```bash
python -m distillation.RAMBO.launch \
  /lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_selection/RAMBO/cases14_500 \
  --cases case14 case500 --counts 100 800
```

This freezes a manifest/source for the existing CPU campaign runner; it does not
submit a job. Outputs use `<root>/<case>/opfdata_envelope/<rambo|uniform>/M<count>/seed42/`.
Run one library with `python -m distillation.RAMBO.run OUTPUT --case case500 --count 800`;
repeat the same command to resume. Incomplete libraries are never published as ready.
Counts have separate identities; runs sharing a seed are not independent replicates.
Follow-up training chooses its own common model, initialization and update budget.
Generation preserves split seed 42, never reads validation/test examples, and keeps
the same feature/target ordering and units as the condensation exports.
CPU regression tests: `python -m pytest distillation/test/test_rambo.py distillation/test/test_rambo_study.py -q`.
The local integration check `~/tests/smoke_rambo_cases.py OUTPUT` exercises real
case14/case500 solves, restart, portable loading and all eight student architectures.

#### Reproducibility
We reimplement RAMBO for OPFData graphs, fixed dataset splits and resumable runs. The bilevel structure, relaxed KKT conditions and final AC-OPF solve follow the paper, with explicit adaptations from the [reference code](https://github.com/SamChevalier/RAMBO/tree/10952cf812bbd378c3e54fc84abf82806a02660a).

- **Different data integration:** A training-only OPFData template supplies the graph format and angle-reference convention; nominal loads come from PGLib. We omit disabled equipment, as OPFData does. Case500 keeps OPFData's reference at bus 313 (PGLib uses 311). Validation/test remain unchanged.

- **Narrower load range:** Defaults are 0.8–1.2 times nominal demand, matching OPFData FullTop's load bounds; the paper uses 0.5–2.0 and the GitHub example 0.5–1.5. Active optimization changes the sampling distribution within those bounds.

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
