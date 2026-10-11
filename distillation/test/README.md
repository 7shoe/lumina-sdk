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

The twelve-XPU continuation check is documented in [HMDC](../HMDC/README.md).
