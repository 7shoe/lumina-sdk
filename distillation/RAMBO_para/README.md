# RAMBO* (`RAMBO_para`)

A separate, batched variant of RAMBO. Existing `distillation/RAMBO` code,
campaigns and reports are unchanged. Use **RAMBO*** in tables and **RAMBO_para**
in metadata; these results must not silently fill the original RAMBO column.

Each batch proposes operating points against one frozen accepted library.
Candidates use the existing relaxed-KKT search, nominal-load bounds `[0.8, 1.2]`,
final AC-OPF solve and independent physical checks. After the batch finishes,
accept valid, nonduplicate points in attempt order, then refresh the library.
The first point is a uniform-load OPF solve. This differs from sequential RAMBO,
which refreshes after each accepted point. Worker completion time never selects data.

`proposal_batch_size` (default 32) is part of the scientific configuration.
`workers` (default 32) controls CPU concurrency only: changing it preserves batch
membership, random seeds and acceptance order. Julia and BLAS use one thread per
worker; the runner pins each worker to a distinct physical core. Hardware and
solver time limits can still affect numerical results. Larger worker counts need
a memory/throughput check; 208 hardware threads are not 208 physical cores.
Concurrency is capped by batch size: to test 64 simultaneous proposals, prepare
a **new** run with `--proposal-batch-size 64` and launch with 64 workers.

Prepare a separate campaign, then submit its frozen launcher:

```bash
root=/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_selection/RAMBO_para/case500_m100_b32
python -m distillation.RAMBO_para.launch "$root" --case case500 --count 100
qsub -v RAMBO_PARA_ROOT="$root",RAMBO_PARA_WORKERS=32 \
  "$root/source/distillation/RAMBO_para/aux/run.pbs"
```

The PBS job uses one node for one hour and stages Julia locally once. Resubmit
the same command to resume. Completed proposal stages are cached; an interrupted
batch retains its original history and ordering. A worker error preserves the
other workers' stages and leaves the run resumable. Incomplete datasets are never
published as complete. No validation/test observations guide generation.

Optional `--bootstrap /path/to/old/RAMBO/dataset.json` copies and rechecks an
existing partial sequential library **into the new run**. Provenance records
the old manifest hash and sample identities. The original directory is read-only;
the new library is a RAMBO* continuation. Bootstrap imports occur only on first
execution; later resumes use the copied samples.

Outputs are in `<root>/run/`: `state.json`, `dataset.json`, per-proposal stage
records, and completed `dataset.pt`, `artifact.json`, `recipe.json`. The existing
`distillation.experiments.datasets.load_dataset(root / "run")` loads these exact
physical inputs and solver targets for matched downstream training. Old dataset
catalogs and evaluations are not modified; add RAMBO* as a separate condition
when a new dataset is ready.

```bash
python -m pytest distillation/test/test_rambo_para.py -q
# Small real Julia/OPFData CPU smoke (new output directory):
python -m distillation.RAMBO_para.aux.smoke /tmp/rambo-para-smoke --case case14
```
