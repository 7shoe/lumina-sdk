## Tests

The benchmark may appear augmentative but the implementation is deep. 
- interweave models and losses of [lumina-bench](https://arxiv.org/abs/2605.02133) properly
- implement selection strategies [BAS](https://arxiv.org/abs/2208.07497) and [RAMBO](https://github.com/SamChevalier/RAMBO) despite lack of or incompatible implementation
- incorporate solver (e.g. [PyPower](https://pypi.org/project/PYPOWER/)) and implement new, sparse solver
- enable synthetic network generation

This directory contains a lot of test cases.

Check if roundabout conversion from OPFdata instance (`HeteroData`) to matrix representation and back works
```bash
python -m pytest distillation/test/test_matrix_conversion.py -v
```

HMDC numerical derivatives, all eight backbone adapters, gradient balancing,
spatial/semantic alignment and checkpoint restoration:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest distillation/test/test_hmdc.py distillation/test/test_dcgm.py -q
```

The twelve-XPU continuation check is documented in [HMDC](../HMDC/aux/README.md).

Independent-launch failure/retry tests and exact HMDC microbatch derivatives:

```bash
python -m pytest distillation/test/test_isolated_launch.py distillation/test/test_hmdc.py -q
```

Inside a two-node PBS allocation, from the frozen runtime directory, run
`python /path/to/isolated_xpu.py --root /lustre/validation/faults`.
It intentionally kills one rank and stalls another, verifies that their sibling
continues on its own 12 XPUs, and checks that retries restore saved state.

RAMBO generation, shared study configuration, resume and portable artifact replay:

```bash
python -m pytest distillation/test/test_rambo.py distillation/test/test_rambo_study.py distillation/test/test_experiment_study.py -q
```

The shared [integration smoke profile](../experiments/configs/README.md) runs real
Julia generation and XPU student evaluation alongside DD/DCGM/HMDC.
