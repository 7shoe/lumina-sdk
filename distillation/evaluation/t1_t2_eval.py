"""Print T1/T2 Markdown tables and save them under reports; no model inference."""
import argparse
from pathlib import Path

if __package__:
    from .utils import build_report, load_campaign, read_source
else:
    from utils import build_report, load_campaign, read_source


T1_RESULTS = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1/results")
T2_RESULTS = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T2/results")
T1_JOBS = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1/jobs")
T2_JOBS = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T2/jobs")
REPORTS = Path("/home/siebenschuh/Projects/lumina-sdk/distillation/evaluation/reports")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    args = parser.parse_args()

    sources = {}
    campaigns = [
        load_campaign("T1", T1_RESULTS, T1_JOBS, args.split, sources),
        load_campaign("T2", T2_RESULTS, T2_JOBS, args.split, sources),
    ]
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("utils.py")):
        read_source(path, sources, "Report generator (local)")
    report = build_report(campaigns, args.split, sources)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"t1_t2_{args.split}.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
