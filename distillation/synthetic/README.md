# Shared synthetic AC-OPF networks

`distillation.synthetic` provides feasible synthetic grids for DD, DCGM and HMDC.
Use `conservative` to edit training networks or `random` to draw fresh topologies
and equipment. Accepted graphs carry freshly checked AC-OPF labels.

```python
from distillation.synthetic import generate

result = generate(
    training_graphs, mode="random", original_buses=30,
    bus_range=(23, 38), count=12, steps=3, seed=42,
)
graphs = result.physical_graphs
```

Supply native OPFData graphs from the training split. Campaign size ranges are
30 → 23–38, 57 → 43–71 and 118 → 89–148 buses; pass `bus_range` explicitly when
calling `generate`. Feasibility checks do not establish global optimality.

The Python needed for benchmark generation and training stays here. External PBS
launchers call `python -m distillation.synthetic.campaign --manifest PATH`.
Normal DD/DCGM/HMDC runs do not import `synthetic.aux`.

[Detailed documentation](aux/README.md), standalone generation, reporting,
plotting, the example PBS script and historical results live in `aux/`.
