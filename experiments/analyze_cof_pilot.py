"""Aggregate corrected CoF pilot runs and official BFCL score artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


CATEGORIES = ("simple_java", "simple_javascript", "multiple")
POLICIES = ("D0", "AC1", "DR1", "COF1", "DR2", "GCOF2")
RESULT_NAMES = {
    "0.6b": "Qwen_Qwen3-0.6B-FC",
    "1.7b": "Qwen_Qwen3-1.7B-FC",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def incorrect_ids(policy_root: Path, result_name: str) -> set[str]:
    score_root = policy_root / "score" / result_name / "non_live"
    incorrect: set[str] = set()
    for category in CATEGORIES:
        rows = load_jsonl(score_root / f"BFCL_v4_{category}_score.json")
        incorrect.update(row["id"] for row in rows[1:])
    return incorrect


def category_for_id(item_id: str) -> str:
    for category in CATEGORIES:
        if item_id.startswith(f"{category}_"):
            return category
    raise ValueError(f"Unknown BFCL item id: {item_id}")


def paired_statistics(
    constrained: dict[str, bool],
    diagnostic: dict[str, bool],
    ids: list[str],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    constrained_values = np.array(
        [constrained[item_id] for item_id in ids],
        dtype=np.int8,
    )
    diagnostic_values = np.array(
        [diagnostic[item_id] for item_id in ids],
        dtype=np.int8,
    )
    differences = constrained_values - diagnostic_values
    helpful = int(np.sum(differences == 1))
    harmful = int(np.sum(differences == -1))
    discordant = helpful + harmful
    sampled_indices = rng.integers(
        0,
        len(ids),
        size=(bootstrap_samples, len(ids)),
    )
    bootstrap_differences = differences[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])
    return {
        "accuracy_difference": float(differences.mean()),
        "accuracy_difference_percentage_points": float(
            differences.mean() * 100
        ),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_95_ci_percentage_points": [
            float(lower * 100),
            float(upper * 100),
        ],
        "helpful": helpful,
        "harmful": harmful,
        "exact_mcnemar_p": (
            float(binomtest(helpful, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "bootstrap_samples": bootstrap_samples,
    }


def analyze_model(
    model_root: Path,
    result_name: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    available = [
        policy
        for policy in POLICIES
        if (model_root / policy / "summary.json").exists()
        and (model_root / policy / "score" / result_name / "non_live").exists()
    ]
    if "D0" not in available:
        raise FileNotFoundError(f"No scored D0 run under {model_root}")

    direct_rows = load_jsonl(model_root / "D0" / "checkpoint.jsonl")
    ids = [row["id"] for row in direct_rows]
    direct_summary = json.loads(
        (model_root / "D0" / "summary.json").read_text(encoding="utf-8")
    )
    incorrect = {
        policy: incorrect_ids(model_root / policy, result_name)
        for policy in available
    }
    correct = {
        policy: {item_id: item_id not in incorrect[policy] for item_id in ids}
        for policy in available
    }

    output: dict[str, Any] = {"n": len(ids), "policies": {}}
    for policy in available:
        policy_root = model_root / policy
        summary = json.loads(
            (policy_root / "summary.json").read_text(encoding="utf-8")
        )
        rows = load_jsonl(policy_root / "checkpoint.jsonl")
        rows_by_id = {row["id"]: row for row in rows}
        correct_count = sum(correct[policy].values())
        added_tokens = (
            summary["total_input_tokens"]
            + summary["total_output_tokens"]
            - direct_summary["total_input_tokens"]
            - direct_summary["total_output_tokens"]
        )
        added_latency = (
            summary["total_latency_seconds"]
            - direct_summary["total_latency_seconds"]
        )
        helpful = sum(
            not correct["D0"][item_id] and correct[policy][item_id]
            for item_id in ids
        )
        harmful = sum(
            correct["D0"][item_id] and not correct[policy][item_id]
            for item_id in ids
        )
        wrong_valid = sum(
            rows_by_id[item_id]["final_schema_valid"]
            and not correct[policy][item_id]
            for item_id in ids
        )
        by_category = {}
        for category in CATEGORIES:
            category_ids = [
                item_id for item_id in ids if category_for_id(item_id) == category
            ]
            by_category[category] = {
                "n": len(category_ids),
                "correct": sum(correct[policy][item_id] for item_id in category_ids),
            }

        output["policies"][policy] = {
            "correct": correct_count,
            "accuracy": correct_count / len(ids),
            "by_category": by_category,
            "final_schema_valid": summary["final_schema_valid"],
            "wrong_valid": wrong_valid,
            "helpful_vs_D0": helpful,
            "harmful_vs_D0": harmful,
            "added_tokens_vs_D0": added_tokens,
            "added_latency_seconds_vs_D0": added_latency,
            "correct_recoveries_per_1000_added_tokens": (
                helpful * 1000 / added_tokens if added_tokens > 0 else None
            ),
        }

    for constrained, diagnostic in (("COF1", "DR1"), ("GCOF2", "DR2")):
        if constrained not in available or diagnostic not in available:
            continue
        output[f"{constrained}_vs_{diagnostic}"] = {
            "net_correct": (
                output["policies"][constrained]["correct"]
                - output["policies"][diagnostic]["correct"]
            ),
            **paired_statistics(
                correct[constrained],
                correct[diagnostic],
                ids,
                rng,
                bootstrap_samples,
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).parent / "cof_pilot228" / "runs",
    )
    parser.add_argument("--model", choices=sorted(RESULT_NAMES), action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    models = args.model or [
        model
        for model in RESULT_NAMES
        if (args.runs_root / model / "D0" / "summary.json").exists()
    ]
    rng = np.random.default_rng(args.seed)
    analysis = {
        model: analyze_model(
            args.runs_root / model,
            RESULT_NAMES[model],
            rng,
            args.bootstrap_samples,
        )
        for model in models
    }
    if len(models) > 1:
        combined: dict[str, Any] = {
            "unit": "BFCL item clustered across models",
            "models": models,
        }
        for constrained, diagnostic in (("COF1", "DR1"), ("GCOF2", "DR2")):
            differences_by_model = []
            shared_ids: list[str] | None = None
            for model in models:
                model_root = args.runs_root / model
                if not all(
                    (
                        model_root
                        / policy
                        / "score"
                        / RESULT_NAMES[model]
                        / "non_live"
                    ).exists()
                    for policy in (constrained, diagnostic)
                ):
                    differences_by_model = []
                    break
                ids = [
                    row["id"]
                    for row in load_jsonl(model_root / "D0" / "checkpoint.jsonl")
                ]
                if shared_ids is None:
                    shared_ids = ids
                elif ids != shared_ids:
                    raise ValueError("Model manifests are not identically ordered")
                constrained_bad = incorrect_ids(
                    model_root / constrained,
                    RESULT_NAMES[model],
                )
                diagnostic_bad = incorrect_ids(
                    model_root / diagnostic,
                    RESULT_NAMES[model],
                )
                differences_by_model.append(
                    np.array(
                        [
                            int(item_id not in constrained_bad)
                            - int(item_id not in diagnostic_bad)
                            for item_id in ids
                        ],
                        dtype=np.int8,
                    )
                )
            if not differences_by_model or shared_ids is None:
                continue
            difference_matrix = np.stack(differences_by_model)
            per_item_mean = difference_matrix.mean(axis=0)
            sampled_indices = rng.integers(
                0,
                len(shared_ids),
                size=(args.bootstrap_samples, len(shared_ids)),
            )
            bootstrap = per_item_mean[sampled_indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            combined[f"{constrained}_vs_{diagnostic}"] = {
                "accuracy_difference": float(difference_matrix.mean()),
                "accuracy_difference_percentage_points": float(
                    difference_matrix.mean() * 100
                ),
                "item_cluster_bootstrap_95_ci": [
                    float(lower),
                    float(upper),
                ],
                "item_cluster_bootstrap_95_ci_percentage_points": [
                    float(lower * 100),
                    float(upper * 100),
                ],
                "bootstrap_samples": args.bootstrap_samples,
            }
        analysis["combined"] = combined
    rendered = json.dumps(analysis, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
