"""Read saved metrics, average seeds equally, and report sample SD (n-1).

OPF error is bus MSE + generator MSE, mixing rad² and p.u.². Violation is
(balance L2 + thermal-excess L2) / sqrt(bus count), mixing p.u. and p.u.².
Cost gap is the saved mean of 100 * (predicted - reference) / reference;
the reference is the dataset solver's generation cost. No physics is rerun.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev


MODELS = ("GCN", "GAT", "GIN", "Transformer", "HGT", "RGAT", "HEAT", "HeteroGNN")
METRICS = ("opf_solution_error", "constraint_violation", "cost_difference_pct")
PRECISION = (6, 4, 3)
ALIASES = {"opf_solution_error": "sdk_mse", "constraint_violation": "violation"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_source(path, sources, origin):
    """Read once and retain the exact input hash for the source list."""
    content = path.read_bytes()
    sources[path] = (origin, hashlib.sha256(content).hexdigest())
    return content.decode("utf-8")


def job_history(jobs, sources):
    """Associate runs with PBS invocations recorded in the saved job logs."""
    submissions = json.loads(read_source(jobs / "submissions.json", sources, "Submission registry"))
    history = defaultdict(list)
    for job in submissions:
        if job["kind"] != "train":
            continue
        text = read_source(Path(job["log"]), sources, f"PBS {job['id']}")
        for run in re.findall(r"rank=\d+ ([^\s]+/seed\d+): exit=0", text):
            if str(job["id"]) not in history[run]:
                history[run].append(str(job["id"]))
    return history


def case_numbers(name):
    return tuple(map(int, re.findall(r"case(\d+)", name)))


def case_label(name):
    return ", ".join(map(str, case_numbers(name)))


def check_result(row, split, audit):
    """Check stored metric identities and counts without loading checkpoints."""
    path, metrics = row["path"], row["metrics"]
    require(row["identity"]["split"] == split, f"Wrong split: {path}")
    require(row["trained_samples"] == audit["samples_per_run"], f"Wrong training budget: {path}")
    require(0 < row["checkpoint_samples"] <= row["trained_samples"], f"Wrong checkpoint count: {path}")
    require(Path(row["checkpoint"]).name == "best.pt", f"Unexpected checkpoint: {path}")
    require(type(metrics["n_samples"]) is int and metrics["n_samples"] > 0, f"Invalid sample count: {path}")
    require(all(math.isfinite(value) for value in metrics.values()), f"Non-finite metrics: {path}")
    nonnegative = ("mse", "sdk_mse", "va_mse", "vm_mse", "pg_mse", "qg_mse",
                   "balance_l2", "thermal_l2", "bounds_l2", *METRICS[:2])
    require(all(metrics[key] >= 0 for key in nonnegative), f"Negative error/residual: {path}")
    require(metrics["reference_cost"] > 0, f"Nonpositive reference cost: {path}")
    expected = {key: metrics[alias] for key, alias in ALIASES.items()}
    expected["sdk_mse"] = sum(metrics[f"{key}_mse"] for key in ("va", "vm", "pg", "qg")) / 2
    if row["evaluated_case"] != "all":
        buses = case_numbers(row["evaluated_case"])[0]
        expected["violation"] = (metrics["balance_l2"] + metrics["thermal_l2"]) / math.sqrt(buses)
    for key, value in expected.items():
        require(math.isclose(metrics[key], value, rel_tol=1e-9, abs_tol=1e-12),
                f"Inconsistent {key}: {path}")
    for key, total in row.get("accumulator", {}).get("sums", {}).items():
        count = row["accumulator"]["counts"][key]
        require(count > 0 and math.isclose(metrics[key], total / count, rel_tol=1e-9, abs_tol=1e-12),
                f"Inconsistent accumulator for {key}: {path}")


def check_collection(total, parts):
    """Verify collection metrics using each metric's saved sample counts."""
    require(total["metrics"]["n_samples"] == sum(p["metrics"]["n_samples"] for p in parts),
            f"Collection sample count mismatch: {total['path']}")
    for part in parts:
        for key in ("specification", "checkpoint_sha256"):
            require(part["identity"][key] == total["identity"][key], f"Mixed checkpoints: {part['path']}")
        require(part["checkpoint_samples"] == total["checkpoint_samples"],
                f"Mixed checkpoint budgets: {part['path']}")
    for metric in METRICS:
        key = ALIASES.get(metric, metric)
        counts = [p["accumulator"]["counts"][key] for p in parts]
        value = sum(p["metrics"][metric] * n for p, n in zip(parts, counts)) / sum(counts)
        require(math.isclose(total["metrics"][metric], value, rel_tol=1e-9, abs_tol=1e-12),
                f"Collection {metric} mismatch: {total['path']}")


def load_campaign(name, results, jobs, split, sources):
    """Load individual result JSONs; summary CSVs are not used as inputs."""
    audit = json.loads(read_source(jobs / "verified.json", sources, "Saved campaign audit"))
    require(audit["campaign"] == name, f"Wrong audit for {name}")
    history = job_history(jobs, sources)
    paths = sorted(results.glob(f"*/*/seed*/{split}.json"))
    require(len(paths) == audit["runs"], f"Missing or extra {name} runs: {len(paths)} vs {audit['runs']}")
    rows, tables = [], defaultdict(list)
    for path in paths:
        model, group, seed_dir = path.relative_to(results).parts[:3]
        run = f"{model}/{group}/{seed_dir}"
        require(run in history, f"No PBS provenance: {run}")
        origin = " → ".join(history[run])
        total = json.loads(read_source(path, sources, origin))
        files = [(path, total)]
        if name == "T2":
            cases = total["identity"]["cases"]
            require(set(total["samples_per_case"]) == set(cases), f"Wrong training cases: {path}")
            require(sum(total["samples_per_case"].values()) == total["trained_samples"],
                    f"Wrong per-case training budget: {path}")
            expected = {path.with_suffix("") / f"{case}.json" for case in cases}
            require(set(path.with_suffix("").glob("*.json")) == expected, f"Missing or extra cases: {path}")
            files += [(p, json.loads(read_source(p, sources, origin))) for p in sorted(expected)]
        for file, row in files:
            require(row["model"] == model and f"seed{row['seed']}" == seed_dir,
                    f"Result/path identity mismatch: {file}")
            require(row.get("collection", row.get("case")) == group, f"Wrong group: {file}")
            case = row.get("case", "all")
            require(name == "T1" or file == path or file.stem == case, f"Wrong case: {file}")
            row.update(path=file, group=group, evaluated_case=case)
            check_result(row, split, audit)
            rows.append(row)
            tables[group, case].append(row)
        if name == "T2":
            check_collection(total, [row for _, row in files[1:]])
    expected = {(model, seed) for model in MODELS for seed in audit["model_seeds"]}
    require(len(audit["model_seeds"]) >= 2, "Sample standard deviation needs multiple seeds")
    for key, group_rows in tables.items():
        require(len(group_rows) == len(expected)
                and {(r["model"], r["seed"]) for r in group_rows} == expected,
                f"Missing, duplicate, or unexpected model/seeds in {name} {key}")
    return {"name": name, "audit": audit, "rows": rows, "tables": tables}


def summarize(rows):
    """Give equal weight to each seed's saved holdout mean; SD uses n-1."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row["metrics"])
    stats = {}
    for model in MODELS:
        stats[model] = {}
        for metric in METRICS:
            values = [row[metric] for row in grouped[model]]
            stats[model][metric] = mean(values), stdev(values)
    return stats


def render_table(title, rows, split):
    stats = summarize(rows)
    seeds = sorted({r["seed"] for r in rows})
    counts = {r["metrics"]["n_samples"] for r in rows}
    require(len(counts) == 1, f"Unequal holdout sizes in {title}")
    best = {key: min(s[key][0] for s in stats.values()) for key in METRICS[:2]}
    lines = [f"### {title}", "", f"{split.capitalize()} means over seeds {', '.join(map(str, seeds))}; "
             f"{next(iter(counts)):,} holdout samples per seed.", "",
             "| Model | OPF solution error (composite score) ↓ | Constraint violation (composite score) ↓ | Cost difference (%) |",
             "|---|---:|---:|---:|"]
    for model, values in stats.items():
        cells = []
        for key, digits in zip(METRICS, PRECISION):
            value = values[key][0]
            cell = f"{value:+.{digits}f}" if key == METRICS[-1] else f"{value:.{digits}f}"
            cells.append(f"**{cell}**" if key in best and value == best[key] else cell)
        lines.append(f"| {model} | {' | '.join(cells)} |")
    ranges = [(min(s[key][1] for s in stats.values()), max(s[key][1] for s in stats.values()))
              for key in METRICS]
    lines += ["", f"Across models, sample standard deviations over the {len(seeds)} seeds ranged from "
              f"**{ranges[0][0]:.6f} to {ranges[0][1]:.6f}** for OPF solution error, "
              f"**{ranges[1][0]:.4f} to {ranges[1][1]:.4f}** for constraint violation, and "
              f"**{ranges[2][0]:.3f} to {ranges[2][1]:.3f} percentage points** for cost difference."]
    return "\n".join(lines)


def smoke_summary(campaigns):
    """Check shared holdouts and describe residuals; this is not a feasibility certificate."""
    holdouts, references = {}, {}
    lines = ["## Consistency and AC-OPF interpretation", ""]
    for campaign in campaigns:
        individual = [r for r in campaign["rows"] if r["evaluated_case"] != "all"]
        for row in individual:
            case = row["evaluated_case"]
            key = campaign["name"], row["group"], case
            identity = (row["identity"]["indices_sha256"], row["metrics"]["n_samples"])
            reference = row["metrics"]["reference_cost"]
            require(holdouts.setdefault(key, identity) == identity, f"Different holdouts for {key}")
            previous = references.setdefault(case, (identity[1], reference))
            require(previous[0] == identity[1] and math.isclose(previous[1], reference, rel_tol=1e-10),
                    f"Different holdout sizes/reference costs for {case}")
        worst = max(individual, key=lambda r: r["metrics"]["constraint_violation"])
        gaps = [r["metrics"]["cost_difference_pct"] for r in individual]
        checkpoints = [r["checkpoint_samples"] for r in individual]
        context = f"collection {{{case_label(worst['group'])}}}, " if campaign["name"] == "T2" else ""
        lines += [f"- **{campaign['name']}**: {len(campaign['rows']):,} result files and "
                  f"{campaign['audit']['runs']:,} runs passed the consistency checks. "
                  f"Selected checkpoints span {min(checkpoints):,}–{max(checkpoints):,} training presentations. "
                  f"Largest per-seed case violation: {worst['metrics']['constraint_violation']:.4f} "
                  f"({worst['model']}, {context}case{case_label(worst['evaluated_case'])}, seed {worst['seed']}). "
                  f"Signed mean cost gaps span {min(gaps):+.3f}% to {max(gaps):+.3f}%; "
                  f"{sum(g < 0 for g in gaps)} of {len(gaps)} per-case seed results have negative gaps."]
    lines += ["", "Checks cover finite/nonnegative errors, metric identities, seed coverage, training budgets, "
              "holdout hashes within each case/collection, reference costs across campaigns, and T2 collection aggregation. "
              "T2 hashes include collection index offsets and cannot be compared directly with T1 hashes. "
              "Large residuals indicate poor constraint satisfaction even when solution MSE is small. "
              "A negative cost gap alone does not establish an improvement over the reference solver. "
              "Signed averages can hide cancellation; these aggregate results do not establish feasibility "
              "for individual samples or measure inference speed."]
    return "\n".join(lines)


def build_report(campaigns, split, sources):
    """Render T1 cases, then T2 collections and their individual cases, then sources."""
    sections = [f"# T1 and T2: {split} results", "Each table averages saved holdout metrics equally across "
                "model seeds. Bold identifies the smallest unrounded mean in the first two columns. "
                "Sample standard deviations use n−1 and appear only in the sentence below each table.",
                "OPF solution error = bus MSE + generator MSE, combining voltage-angle errors (rad²) "
                "with voltage-magnitude and power errors (p.u.²). Constraint violation = "
                "(power-balance L2 + thermal-excess L2) / √(number of buses); thermal excess uses "
                "max(|S|² − limit², 0) at both branch ends. It combines p.u. and p.u.², so both "
                "columns are composite scores. Here p.u. means per unit; bound residuals are separate diagnostics.",
                "Cost difference is the saved per-sample mean of 100 × (predicted generation cost − "
                "reference generation cost) / reference generation cost, then averaged across seeds. "
                "The reference is the dataset's AC-OPF solver solution. Results evaluate the "
                "validation-selected best.pt; its training count can precede the completed budget."]
    for campaign in campaigns:
        name, audit = campaign["name"], campaign["audit"]
        sections += [f"## {name}", f"{audit['runs']:,} runs; {audit['samples_per_run']:,} training "
                     f"presentations per run; split seed {audit['split_seed']}. "
                     + ("T2's budget is shared across the cases in each collection." if name == "T2" else "")]
        for (group, case), rows in sorted(campaign["tables"].items(),
                                         key=lambda item: (case_numbers(item[0][0]), case_numbers(item[0][1]))):
            title = f"T1 — case{case_label(case)}" if name == "T1" else (
                f"T2 — collection {{{case_label(group)}}}: "
                + ("all cases" if case == "all" else f"case{case_label(case)}"))
            sections.append(render_table(title, rows, split))
    sections.append(smoke_summary(campaigns))
    sections += ["## SOURCES", "Metrics come directly from the result JSONs below. SHA-256 identifies "
                 "the exact bytes read. PBS history comes from submissions.json and matching run/exit lines "
                 "in the saved training logs. Arrows show successive recorded invocations, including "
                 "resumes; the last entry is the last recorded successful invocation. These logs do not "
                 "identify the exact writing job for every file within a resumed evaluation. "
                 "The audit and this report were produced by local Python processes.",
                 "| File analyzed | PBS job history / role | SHA-256 |\n|---|---|---|\n"
                 + "\n".join(f"| [{path}]({path.as_posix()}) | {origin} | `{digest}` |"
                              for path, (origin, digest) in sorted(sources.items()))]
    return "\n\n".join(sections) + "\n"
