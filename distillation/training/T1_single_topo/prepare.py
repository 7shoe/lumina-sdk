"""Prepare canonical graph caches once; repeated launches skip finished groups."""
import argparse
import time

from .common import DEFAULT_CONFIG, load_config
from .data import GROUPS, finish_case, prepare_group
from .state import Stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--seconds", type=float, default=3300)
    args = parser.parse_args()
    rank, size, comm = 0, 1, None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    config = load_config(args.config)
    tasks = [(case, group) for case in config["cases"] for group in range(GROUPS)]
    failures = 0
    with Stop(args.seconds) as stop:
        for case, group in tasks[rank::size]:
            if stop():
                break
            started = time.monotonic()
            try:
                ready = prepare_group(config["paths"]["raw"], config["paths"]["data"], case, group)
                if ready:
                    finish_case(config["paths"]["data"], case)
                print(f"rank={rank} {case}/group_{group}: {'ready' if ready else 'busy'} "
                      f"({time.monotonic() - started:.1f}s)", flush=True)
            except Exception as exc:
                failures += 1
                print(f"FAILED {case}/group_{group}: {exc}", flush=True)
    if comm is not None:
        failures = comm.allreduce(failures)
    if rank == 0:
        for case in config["cases"]:
            try:
                ready = finish_case(config["paths"]["data"], case)
            except (OSError, ValueError) as exc:
                print(f"FAILED {case}: {exc}", flush=True)
                ready = False
            failures += int(not ready)
            print(f"{case}: {'ready' if ready else 'incomplete'}")
    if comm is not None:
        failures = comm.bcast(failures, root=0)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
