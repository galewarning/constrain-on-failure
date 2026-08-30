"""Analyze the subsequent DR1-P component ablation on the frozen live-300 sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_cof_pilot import load_jsonl, paired_statistics
from analyze_cof_robustness import clustered_comparison


CATEGORIES = ("live_simple", "live_multiple")
POLICIES = ("DR1", "DR1-P", "COF1")
RESULT_NAMES = {
    "0.6b": "Qwen_Qwen3-0.6B-FC",
    "1.7b": "Qwen_Qwen3-1.7B-FC",
    "granite": "ibm-granite_granite-3.2-8b-instruct",
}


def incorrect_ids(policy_root: Path, result_name: str) -> set[str]:
    score_root = policy_root / "score" / result_name / "live"
    incorrect: set[str] = set()
    for category in CATEGORIES:
        rows = load_jsonl(score_root / f"BFCL_v4_{category}_score.json")
        incorrect.update(row["id"] for row in rows[1:])
    return incorrect


def policy_root(
    frozen_runs: Path,
    ablation_runs: Path,
    model: str,
    policy: str,
) -> Path:
    root = ablation_runs if policy == "DR1-P" else frozen_runs
    return root / model / policy


def analyze_model(
    frozen_runs: Path,
    ablation_runs: Path,
    model: str,
    result_name: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, bool]]]:
    rows = {
        policy: load_jsonl(
            policy_root(frozen_runs, ablation_runs, model, policy)
            / "checkpoint.jsonl"
        )
        for policy in POLICIES
    }
    ids = [row["id"] for row in rows["DR1"]]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError(f"Expected 300 unique IDs for {model}")
    for policy in POLICIES:
        if [row["id"] for row in rows[policy]] != ids:
            raise ValueError(f"ID order mismatch for {model}/{policy}")

    by_id = {
        policy: {row["id"]: row for row in rows[policy]}
        for policy in POLICIES
    }
    incorrect = {
        policy: incorrect_ids(
            policy_root(frozen_runs, ablation_runs, model, policy),
            result_name,
        )
        for policy in POLICIES
    }
    correct = {
        policy: {item_id: item_id not in incorrect[policy] for item_id in ids}
        for policy in POLICIES
    }

    output: dict[str, Any] = {
        "n": len(ids),
        "status": "subsequent_post_hoc_component_ablation",
        "policies": {},
    }
    for policy in POLICIES:
        root = policy_root(frozen_runs, ablation_runs, model, policy)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        correct_count = sum(correct[policy].values())
        output["policies"][policy] = {
            "correct": correct_count,
            "accuracy": correct_count / len(ids),
            "final_schema_valid": int(summary["final_schema_valid"]),
            "wrong_valid": sum(
                bool(by_id[policy][item_id]["final_schema_valid"])
                and not correct[policy][item_id]
                for item_id in ids
            ),
            "retry_count": int(summary["repairs_attempted"]),
            "total_input_tokens": int(summary["total_input_tokens"]),
            "total_output_tokens": int(summary["total_output_tokens"]),
            "total_tokens": int(summary["total_input_tokens"])
            + int(summary["total_output_tokens"]),
            "total_latency_seconds": float(summary["total_latency_seconds"]),
        }

    for candidate, baseline in (("DR1-P", "DR1"), ("COF1", "DR1-P"), ("COF1", "DR1")):
        comparison = paired_statistics(
            correct[candidate],
            correct[baseline],
            ids,
            rng,
            bootstrap_samples,
        )
        comparison["discordant"] = comparison["helpful"] + comparison["harmful"]
        comparison["status"] = (
            "original_bundle_comparison"
            if (candidate, baseline) == ("COF1", "DR1")
            else "secondary_post_hoc_component_comparison"
        )
        output[f"{candidate}_vs_{baseline}"] = comparison

    invalid_ids = [
        item_id
        for item_id in ids
        if not bool(by_id["DR1"][item_id]["initial_schema_valid"])
    ]
    conditional: dict[str, Any] = {"n": len(invalid_ids), "policies": {}}
    for policy in POLICIES:
        cells = {
            "validator_valid_bfcl_correct": 0,
            "validator_valid_bfcl_incorrect": 0,
            "validator_invalid_bfcl_correct": 0,
            "validator_invalid_bfcl_incorrect": 0,
        }
        for item_id in invalid_ids:
            valid = bool(by_id[policy][item_id]["final_schema_valid"])
            is_correct = bool(correct[policy][item_id])
            key = (
                "validator_valid" if valid else "validator_invalid"
            ) + ("_bfcl_correct" if is_correct else "_bfcl_incorrect")
            cells[key] += 1
        conditional["policies"][policy] = cells
    output["conditional_on_d0_invalid"] = conditional
    return output, ids, correct


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent
    parser.add_argument(
        "--frozen-runs",
        type=Path,
        default=base / "cof_confirmatory_live300" / "runs",
    )
    parser.add_argument(
        "--ablation-runs",
        type=Path,
        default=base / "cof_component_ablation_dr1p" / "runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "cof_component_ablation_dr1p" / "analysis.json",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    analysis: dict[str, Any] = {
        "status": "subsequent_post_hoc_component_ablation",
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    correctness: dict[str, dict[str, dict[str, bool]]] = {}
    shared_ids: list[str] | None = None
    for model, result_name in RESULT_NAMES.items():
        model_output, ids, correct = analyze_model(
            args.frozen_runs.resolve(),
            args.ablation_runs.resolve(),
            model,
            result_name,
            rng,
            args.bootstrap_samples,
        )
        analysis[model] = model_output
        correctness[model] = correct
        if shared_ids is None:
            shared_ids = ids
        elif ids != shared_ids:
            raise ValueError("Model manifests are not identically ordered")

    assert shared_ids is not None
    comparisons = (("DR1-P", "DR1"), ("COF1", "DR1-P"))
    analysis["combined"] = {}
    for candidate, baseline in comparisons:
        analysis["combined"][f"qwen_two_model_{candidate}_vs_{baseline}"] = clustered_comparison(
            correctness,
            ["0.6b", "1.7b"],
            shared_ids,
            candidate,
            baseline,
            rng,
            args.bootstrap_samples,
        )
        analysis["combined"][f"three_model_{candidate}_vs_{baseline}"] = clustered_comparison(
            correctness,
            ["0.6b", "1.7b", "granite"],
            shared_ids,
            candidate,
            baseline,
            rng,
            args.bootstrap_samples,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
