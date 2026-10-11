## Baseline: "Bucketized Active Sampling for Learning ACOPF"

Authors: Michael Klamkin, Mathieu Tanneau, Terrence W.K. Mak, Pascal Van Hentenryck
Year: 


#### Reproducibility
There does not appear to be an implementation. this is an honest attempt to make a straightforward, pragmatic baseline.
Some change come from us. E.g., the use of Deepminds [`OPFData`](https://arxiv.org/abs/2406.07234v1) and our train/val/test split remains pre-defined by our code. 


- **Existing samples:** We select already-solved OPFData examples. Their paper generates new examples and solves AC-OPF for them, so we cannot evaluate its solver-time savings or avoidance of infeasible proposals.

- **Different buckets:** We group samples by total active demand, using quantiles by default. The paper uses a regional load multiplier that OPFData does not provide.

- **Scoring uses training data:** We reserve part of the training split to score buckets. Validation and test sets remain separate for benchmarking.

- **Different models and outputs:** We use Lumina’s graph models and an MLP to predict bus and generator quantities. These differ from the paper’s architecture and outputs.

- **Some settings are our choices:** We explicitly choose learning-rate, patience, and acquisition settings where details are missing. Where the pseudocode is ambiguous about when to acquire samples, we follow the paper’s text.

- **Different experiments:** We use smaller grids and optionally train new models on the selected data. This differs from the paper’s large-grid experiments, which run for 150 minutes and average results over 50 trials.



#### Reference
```bash
@article{klamkin2024bucketized,
  title={Bucketized active sampling for learning acopf},
  author={Klamkin, Michael and Tanneau, Mathieu and Mak, Terrence WK and Van Hentenryck, Pascal},
  journal={Electric Power Systems Research},
  volume={235},
  pages={110697},
  year={2024},
  publisher={Elsevier}
}
```