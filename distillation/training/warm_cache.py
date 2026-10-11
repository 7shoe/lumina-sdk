"""Read prepared groups sequentially before workers access them at random."""
import argparse
import json
from pathlib import Path
import socket
import time

import yaml

from distillation.training.T1_single_topo.common import canonical_case, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    started = time.monotonic()
    total = 0
    for case in config['cases']:
        folder = Path(config['paths']['data']) / canonical_case(case)
        manifest = read_json(folder / 'manifest.json')
        for record in manifest['groups']:
            path = folder / f"group_{record['group']}.pt"
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
                raise ValueError(f'Prepared cache changed: {path}')
            with path.open('rb') as stream:
                while block := stream.read(8 * 1024 * 1024):
                    total += len(block)
    print(json.dumps({'host': socket.gethostname(), 'cache_read_gib': round(total / 2**30, 2),
                      'seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    main()
