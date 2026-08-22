from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baselines import get_system
from .collector import COLLECTOR_VERSION, prepare_run_directory
from .evaluators import evaluate_scenario
from .metrics import aggregate_repetitions, compute_headline_metrics
from .schema import file_sha256, load_scenarios


EVALUATOR_VERSION = "a-plus-final-v2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("case_id"):
                raise ValueError(f"Trajectory {path}:{line_no} must contain case_id")
            rows.append(value)
    return rows


def _validate_trajectory_coverage(cases, trajectories: list[dict[str, Any]], repetitions: int, *, system: str = "full") -> None:
    expected = {(case.id, repetition, system) for case in cases for repetition in range(1, repetitions + 1)}
    actual = {(str(row["case_id"]), int(row.get("repetition", 1)), str(row.get("system", system))) for row in trajectories}
    if len(actual) != len(trajectories):
        raise ValueError("Trajectory input contains duplicate case_id/repetition rows")
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            "Trajectory coverage does not match the selected protocol: "
            f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
        )


def run_evaluation(
    cases,
    trajectories,
    *,
    system: str = "full",
    ablation: str | None = None,
    baseline_hitl_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {case.id: case for case in cases}
    output = []
    for row in trajectories:
        case = by_id.get(str(row["case_id"]))
        if case is None:
            raise ValueError(f"Trajectory references unknown case: {row['case_id']}")
        result = evaluate_scenario(case, row.get("trajectory", row), system=system, ablation=ablation)
        result["repetition"] = int(row.get("repetition", 1))
        result["ground_truth_policy"] = case.expected_policy
        result["expected_unsafe_case"] = case.effective_risk_class in {"HITL_REQUIRED", "DENY_REQUIRED"}
        output.append(result)
    run_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        run_groups[int(row["repetition"])].append(row)
    run_metrics = [
        compute_headline_metrics(group, baseline_hitl_count=baseline_hitl_count)
        for _, group in sorted(run_groups.items())
    ]
    metrics = compute_headline_metrics(output, baseline_hitl_count=baseline_hitl_count)
    metrics["repetitions"] = aggregate_repetitions(run_metrics)
    return output, metrics


def build_manifest(dataset: str | Path, *, system: str, model: str, repetitions: int, cases: list[Any], status: str, run_id: str | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset)
    benchmark_root = dataset_path.parent
    test_path = benchmark_root / "test.jsonl"
    dev_path = benchmark_root / "dev.jsonl"
    safety_path = benchmark_root / "safety_cases.jsonl"
    return {
        "benchmark_version": EVALUATOR_VERSION,
        "status": status,
        "run_id": run_id,
        "git_commit": _git_commit(),
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "dev_sha256": file_sha256(dev_path) if dev_path.exists() else None,
        "test_sha256": file_sha256(test_path) if test_path.exists() else None,
        "safety_sha256": file_sha256(safety_path) if safety_path.exists() else None,
        "model": model,
        "provider": "Alibaba Bailian",
        "model_parameters": {},
        "system": get_system(system).name,
        "repetitions": repetitions,
        "scenario_count": len(cases),
        "estimated_model_attempts": len(cases) * repetitions,
        "full_planned_attempts": 36 * repetitions,
        "all_systems_planned_attempts": 36 * repetitions * 3,
        "best_of_n": False,
        "paid_usage": False,
        "free_tier_only": True,
        "collector_version": COLLECTOR_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved A+ trajectories without an LLM judge.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system", choices=("naive_tool", "hitl_only", "full"), default="full")
    parser.add_argument("--model", default="qwen-plus-2025-07-28")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--input", help="JSONL trajectories; omit for validation/estimation only")
    parser.add_argument("--output")
    parser.add_argument("--run-id", help="Immutable formal run id; writes under eval/final/results/<run-id>")
    parser.add_argument("--output-root", default="eval/final/results")
    parser.add_argument("--baseline-input", help="Optional saved baseline trajectories for HITL reduction")
    parser.add_argument("--cases", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ablation", choices=(None, "no_goal_verification", "no_evidence_provenance", "no_atomic_authorization"), default=None)
    args = parser.parse_args()
    cases = load_scenarios(args.dataset)
    selected = {item.strip() for item in args.cases.split(",") if item.strip()}
    if selected:
        cases = [case for case in cases if case.id in selected]
    if args.category:
        cases = [case for case in cases if case.category == args.category]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if args.run_id and args.output:
        parser.error("use either --run-id or --output, not both")
    if not args.run_id and not args.output:
        parser.error("one of --run-id or --output is required")
    output_dir = prepare_run_directory(args.output_root, args.run_id) if args.run_id else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _load_trajectories(args.input) if args.input else []
    status = "EVALUATED" if args.input else "READY_NOT_RUN"
    manifest = build_manifest(args.dataset, system=args.system, model=args.model, repetitions=args.repetitions, cases=cases, status=status, run_id=args.run_id)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not trajectories:
        (output_dir / "summary.json").write_text(json.dumps({"status": status, "manifest": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "scenario_count": len(cases), "estimated_model_attempts": manifest["estimated_model_attempts"]}, ensure_ascii=False))
        return 0
    _validate_trajectory_coverage(cases, trajectories, args.repetitions, system=args.system)
    baseline_hitl_count = None
    if args.baseline_input:
        baseline_trajectories = _load_trajectories(args.baseline_input)
        _validate_trajectory_coverage(cases, baseline_trajectories, args.repetitions, system="hitl_only")
        _, baseline_metrics = run_evaluation(cases, baseline_trajectories, system="hitl_only")
        baseline_hitl_count = int(baseline_metrics["hitl_count"])
    rows, metrics = run_evaluation(
        cases,
        trajectories,
        system=args.system,
        ablation=args.ablation,
        baseline_hitl_count=baseline_hitl_count,
    )
    if any(row.get("attempt_status") in {"ERROR", "BLOCKED"} for row in rows):
        status = "INCOMPLETE_ATTEMPTS"
        manifest["status"] = status
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trajectories.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    summary = {"status": status, "manifest": manifest, "metrics": metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
