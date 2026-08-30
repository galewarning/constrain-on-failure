"""Analyze the three-model CoF robustness extension and end-to-end cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_cof_pilot import load_jsonl, paired_statistics


CATEGORIES = ("live_simple", "live_multiple")
POLICIES = ("D0", "AC1", "DR1", "COF1")
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


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def analyze_model(
    model_root: Path,
    result_name: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, bool]]]:
    rows = {
        policy: load_jsonl(model_root / policy / "checkpoint.jsonl")
        for policy in POLICIES
    }
    ids = [row["id"] for row in rows["D0"]]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError(f"Expected 300 unique D0 IDs under {model_root}")
    for policy in POLICIES:
        if [row["id"] for row in rows[policy]] != ids:
            raise ValueError(f"ID order mismatch for {model_root.name}/{policy}")

    by_id = {
        policy: {row["id"]: row for row in rows[policy]}
        for policy in POLICIES
    }
    incorrect = {
        policy: incorrect_ids(model_root / policy, result_name)
        for policy in POLICIES
    }
    correct = {
        policy: {item_id: item_id not in incorrect[policy] for item_id in ids}
        for policy in POLICIES
    }
    summaries = {
        policy: json.loads(
            (model_root / policy / "summary.json").read_text(encoding="utf-8")
        )
        for policy in POLICIES
    }
    d0_total_tokens = (
        summaries["D0"]["total_input_tokens"]
        + summaries["D0"]["total_output_tokens"]
    )
    d0_correct = sum(correct["D0"].values())

    output: dict[str, Any] = {"n": len(ids), "policies": {}}
    for policy in POLICIES:
        summary = summaries[policy]
        total_tokens = (
            summary["total_input_tokens"] + summary["total_output_tokens"]
        )
        latencies = [float(row["latency"]) for row in rows[policy]]
        correct_count = sum(correct[policy].values())
        added_correct = correct_count - d0_correct
        added_tokens = total_tokens - d0_total_tokens
        added_seconds = (
            float(summary["total_latency_seconds"])
            - float(summaries["D0"]["total_latency_seconds"])
        )
        output["policies"][policy] = {
            "correct": correct_count,
            "accuracy": correct_count / len(ids),
            "by_category": {
                category: {
                    "n": sum(
                        item_id.startswith(f"{category}_") for item_id in ids
                    ),
                    "correct": sum(
                        correct[policy][item_id]
                        for item_id in ids
                        if item_id.startswith(f"{category}_")
                    ),
                }
                for category in CATEGORIES
            },
            "final_schema_valid": int(summary["final_schema_valid"]),
            "wrong_valid": sum(
                bool(by_id[policy][item_id]["final_schema_valid"])
                and not correct[policy][item_id]
                for item_id in ids
            ),
            "retry_count": int(summary["repairs_attempted"]),
            "retry_rate": float(summary["repairs_attempted"]) / len(ids),
            "total_input_tokens": int(summary["total_input_tokens"]),
            "total_output_tokens": int(summary["total_output_tokens"]),
            "total_tokens": int(total_tokens),
            "mean_tokens_per_request": total_tokens / len(ids),
            "total_latency_seconds": float(summary["total_latency_seconds"]),
            "mean_latency_seconds": float(np.mean(latencies)),
            "median_latency_seconds": percentile(latencies, 0.5),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "added_correct_vs_D0": added_correct,
            "added_tokens_vs_D0": added_tokens,
            "added_seconds_vs_D0": added_seconds,
            "added_tokens_per_added_correct_vs_D0": (
                added_tokens / added_correct if added_correct > 0 else None
            ),
            "added_seconds_per_added_correct_vs_D0": (
                added_seconds / added_correct if added_correct > 0 else None
            ),
            "cost_effectiveness_vs_D0": (
                "defined" if added_correct > 0 else "dominated_or_undefined"
            ),
        }

    for candidate, baseline in (("COF1", "DR1"), ("COF1", "AC1")):
        stats = paired_statistics(
            correct[candidate],
            correct[baseline],
            ids,
            rng,
            bootstrap_samples,
        )
        candidate_cost = output["policies"][candidate]
        baseline_cost = output["policies"][baseline]
        added_correct = (
            candidate_cost["correct"] - baseline_cost["correct"]
        )
        added_tokens = (
            candidate_cost["total_tokens"] - baseline_cost["total_tokens"]
        )
        added_seconds = (
            candidate_cost["total_latency_seconds"]
            - baseline_cost["total_latency_seconds"]
        )
        output[f"{candidate}_vs_{baseline}"] = {
            **stats,
            "added_correct": added_correct,
            "added_tokens": added_tokens,
            "added_seconds": added_seconds,
            "added_tokens_per_added_correct": (
                added_tokens / added_correct if added_correct > 0 else None
            ),
            "added_seconds_per_added_correct": (
                added_seconds / added_correct if added_correct > 0 else None
            ),
            "cost_effectiveness": (
                "defined" if added_correct > 0 else "dominated_or_undefined"
            ),
        }

    invalid_ids = [
        item_id
        for item_id in ids
        if not bool(by_id["D0"][item_id]["initial_schema_valid"])
    ]
    conditional: dict[str, Any] = {
        "d0_invalid_n": len(invalid_ids),
        "policies": {},
    }
    for policy in POLICIES:
        states = {
            "valid_correct": 0,
            "valid_incorrect": 0,
            "invalid_correct": 0,
            "invalid_incorrect": 0,
        }
        for item_id in invalid_ids:
            valid = bool(by_id[policy][item_id]["final_schema_valid"])
            is_correct = bool(correct[policy][item_id])
            key = (
                ("valid" if valid else "invalid")
                + "_"
                + ("correct" if is_correct else "incorrect")
            )
            states[key] += 1
        conditional["policies"][policy] = states

    error_classes = sorted(
        {
            by_id["D0"][item_id]["initial_error_class"]
            for item_id in invalid_ids
        }
    )
    conditional["by_initial_error_class"] = {}
    for error_class in error_classes:
        class_ids = [
            item_id
            for item_id in invalid_ids
            if by_id["D0"][item_id]["initial_error_class"] == error_class
        ]
        conditional["by_initial_error_class"][error_class] = {
            "n": len(class_ids),
            **{
                f"{policy}_correct": sum(
                    correct[policy][item_id] for item_id in class_ids
                )
                for policy in POLICIES
            },
        }
    conditional["COF1_vs_DR1"] = {
        "helpful": sum(
            not correct["DR1"][item_id] and correct["COF1"][item_id]
            for item_id in invalid_ids
        ),
        "harmful": sum(
            correct["DR1"][item_id] and not correct["COF1"][item_id]
            for item_id in invalid_ids
        ),
        "ties": sum(
            correct["DR1"][item_id] == correct["COF1"][item_id]
            for item_id in invalid_ids
        ),
    }
    output["conditional_on_invalid_D0"] = conditional
    return output, ids, correct


def clustered_comparison(
    correctness: dict[str, dict[str, dict[str, bool]]],
    models: list[str],
    ids: list[str],
    candidate: str,
    baseline: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    matrix = np.stack(
        [
            np.asarray(
                [
                    int(correctness[model][candidate][item_id])
                    - int(correctness[model][baseline][item_id])
                    for item_id in ids
                ],
                dtype=np.int8,
            )
            for model in models
        ]
    )
    per_item = matrix.mean(axis=0)
    sampled = rng.integers(
        0,
        len(ids),
        size=(bootstrap_samples, len(ids)),
    )
    bootstrap = per_item[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "models": models,
        "unit": "BFCL item clustered across models",
        "accuracy_difference": float(matrix.mean()),
        "accuracy_difference_percentage_points": float(matrix.mean() * 100),
        "item_cluster_bootstrap_95_ci": [float(lower), float(upper)],
        "item_cluster_bootstrap_95_ci_percentage_points": [
            float(lower * 100),
            float(upper * 100),
        ],
        "per_model_difference_percentage_points": {
            model: float(matrix[index].mean() * 100)
            for index, model in enumerate(models)
        },
        "positive_models": int(
            sum(matrix[index].mean() > 0 for index in range(len(models)))
        ),
        "negative_models": int(
            sum(matrix[index].mean() < 0 for index in range(len(models)))
        ),
        "zero_models": int(
            sum(matrix[index].mean() == 0 for index in range(len(models)))
        ),
        "bootstrap_samples": bootstrap_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).parent / "cof_confirmatory_live300" / "runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    analysis: dict[str, Any] = {}
    correctness: dict[str, dict[str, dict[str, bool]]] = {}
    shared_ids: list[str] | None = None
    for model, result_name in RESULT_NAMES.items():
        model_output, ids, correct = analyze_model(
            args.runs_root / model,
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
    all_models = list(RESULT_NAMES)
    qwen_models = ["0.6b", "1.7b"]
    analysis["combined"] = {
        "qwen_two_model_COF1_vs_DR1": clustered_comparison(
            correctness,
            qwen_models,
            shared_ids,
            "COF1",
            "DR1",
            rng,
            args.bootstrap_samples,
        ),
        "three_model_COF1_vs_DR1": clustered_comparison(
            correctness,
            all_models,
            shared_ids,
            "COF1",
            "DR1",
            rng,
            args.bootstrap_samples,
        ),
        "three_model_COF1_vs_AC1": clustered_comparison(
            correctness,
            all_models,
            shared_ids,
            "COF1",
            "AC1",
            rng,
            args.bootstrap_samples,
        ),
    }
    three_model = analysis["combined"]["three_model_COF1_vs_DR1"]
    analysis["robustness_interpretation"] = {
        "cross_family_direction_consistent": three_model["negative_models"] == 0,
        "granite_replicated_positive_direction": (
            analysis["granite"]["COF1_vs_DR1"]["accuracy_difference"] > 0
        ),
        "classification": "heterogeneous_cross_family_result",
        "manuscript_action": (
            "retain the preregistered two-Qwen confirmatory estimate; report "
            "Granite as a negative robustness result and narrow generalization"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
