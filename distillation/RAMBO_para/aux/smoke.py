"""Three-point real Julia CPU smoke: parallel search, resume and portable labels."""
import argparse
import json
from pathlib import Path

import torch

from distillation.experiments.datasets import load_dataset
from distillation.synthetic.learning import benchmark_graph
from distillation.training.T1_single_topo.common import file_hash, write_json
from distillation.training.T1_single_topo.state import Stop
from ..run import configuration, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', default='case14')
    parser.add_argument('--seconds', type=float, default=900)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = configuration(case=args.case, count=3, proposal_batch_size=2, max_attempts=20)
    with Stop(args.seconds) as stop:
        first = run(cfg, args.output, workers=2, stop=stop, session_batches=1)
        assert first['status'] == 'paused' and first['accepted'] == 1, first
        result = run(cfg, args.output, workers=2, stop=stop)
    assert result['status'] == 'complete', result
    loaded = load_dataset(args.output)
    assert loaded['metadata']['method'] == 'RAMBO_para'
    assert len(loaded['graphs']) == 3
    payload = torch.load(args.output / 'dataset.pt', weights_only=False, map_location='cpu')
    for actual, physical in zip(loaded['graphs'], payload['physical_solved_graphs']):
        expected = benchmark_graph(physical, labels=True)
        for kind in actual.node_types:
            torch.testing.assert_close(actual[kind].x, expected[kind].x, rtol=0, atol=0)
        for kind in ('bus', 'generator'):
            torch.testing.assert_close(actual[kind].y, physical[kind].y, rtol=0, atol=0)
    before = file_hash(args.output / 'dataset.pt')
    assert run(cfg, args.output, workers=1) == result
    assert file_hash(args.output / 'dataset.pt') == before
    report = dict(status='PASS', case=args.case, count=3, parallel_workers=2, restart=True,
                  portable_labels=True, result=result)
    write_json(args.output / 'smoke.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
