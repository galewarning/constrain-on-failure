"""Stage 4 sensitivity and complete DR1/COF1 discordance extraction.

This script uses only retained experiment and BFCL artifacts. It does not run
models or alter any frozen experiment output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


MODELS = {
    "qwen3_0.6b": ("0.6b", "Qwen_Qwen3-0.6B-FC"),
    "qwen3_1.7b": ("1.7b", "Qwen_Qwen3-1.7B-FC"),
    "granite_3.3_2b": ("granite", "ibm-granite_granite-3.2-8b-instruct"),
}
CATEGORIES = ("live_simple", "live_multiple")
HARMFUL_CLASSIFICATION = {
    ("qwen3_0.6b", "live_simple_105-62-0"): (
        "schema_permissive_semantic_regression",
        "The schema required both arrays but did not prevent assigning account-link "
        "queries to the greeting array; COF1 was valid but over-assigned values.",
    ),
    ("qwen3_0.6b", "live_multiple_395-138-3"): (
        "schema_permissive_semantic_regression",
        "The string schema did not enforce the documented City, State format; COF1 "
        "omitted the accepted state abbreviation while DR1 retained it.",
    ),
    ("qwen3_1.7b", "live_simple_129-83-1"): (
        "schema_description_default_bias",
        "COF1 retained the {fabricName} placeholder shown in the schema description "
        "instead of substituting the requested fabric in the accepted URL.",
    ),
    ("granite_3.3_2b", "live_simple_145-95-2"): (
        "schema_ground_truth_conflict",
        "The schema enum excluded N/A although its own default and the BFCL accepted "
        "answer allowed N/A, forcing a different enum value.",
    ),
    ("granite_3.3_2b", "live_simple_185-110-0"): (
        "schema_ground_truth_conflict",
        "The schema declared nullable location defaults as strings while BFCL "
        "accepted null, forcing semantically unwanted strings.",
    ),
    ("granite_3.3_2b", "live_multiple_748-169-3"): (
        "schema_underdetermination_semantic_regression",
        "Both Action and Thriller were schema-legal, but only Action matched the "
        "request and BFCL accepted answer.",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def incorrect_ids(model_root: Path, result_name: str, policy: str) -> set[str]:
    root = model_root / policy / "score" / result_name / "live"
    output: set[str] = set()
    for category in CATEGORIES:
        rows = load_jsonl(root / f"BFCL_v4_{category}_score.json")
        output.update(row["id"] for row in rows[1:])
    return output


def context_key(item_id: str) -> str:
    prefix, triplet = item_id.rsplit("_", 1)
    middle = triplet.split("-")[1]
    return f"{prefix}:{middle}"


def bootstrap_ci(
    matrix: np.ndarray,
    ids: list[str],
    rng: np.random.Generator,
    samples: int,
    mode: str,
) -> list[float]:
    if mode == "stratified_item":
        strata = [
            np.asarray(
                [index for index, item_id in enumerate(ids) if item_id.startswith(category)],
                dtype=int,
            )
            for category in CATEGORIES
        ]
        draws = np.empty(samples, dtype=float)
        for sample_index in range(samples):
            selected = np.concatenate(
                [rng.choice(stratum, len(stratum), replace=True) for stratum in strata]
            )
            draws[sample_index] = matrix[:, selected].mean()
    elif mode == "context_cluster":
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, item_id in enumerate(ids):
            grouped[context_key(item_id)].append(index)
        clusters = list(grouped.values())
        draws = np.empty(samples, dtype=float)
        for sample_index in range(samples):
            selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
            selected = np.concatenate(
                [np.asarray(clusters[index], dtype=int) for index in selected_clusters]
            )
            draws[sample_index] = matrix[:, selected].mean()
    else:
        raise ValueError(mode)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return [float(lower * 100), float(upper * 100)]


def simulated_mcnemar_power(
    rng: np.random.Generator,
    n: int,
    discordance_rate: float,
    risk_difference: float,
    simulations: int,
) -> float | None:
    if risk_difference > discordance_rate:
        return None
    helpful_probability = 0.5 * (1 + risk_difference / discordance_rate)
    rejections = 0
    for _ in range(simulations):
        discordant = int(rng.binomial(n, discordance_rate))
        if discordant == 0:
            continue
        helpful = int(rng.binomial(discordant, helpful_probability))
        harmful = discordant - helpful
        if binomtest(min(helpful, harmful), discordant, 0.5).pvalue < 0.05:
            rejections += 1
    return rejections / simulations


def bfcl_records(data_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        tasks = {
            row["id"]: row
            for row in load_jsonl(data_root / f"BFCL_v4_{category}.json")
        }
        answers = {
            row["id"]: row
            for row in load_jsonl(
                data_root / "possible_answer" / f"BFCL_v4_{category}.json"
            )
        }
        for item_id, task in tasks.items():
            records[item_id] = {
                "question": task["question"],
                "function": task["function"],
                "ground_truth": answers[item_id]["ground_truth"],
            }
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).parent / "cof_confirmatory_live300" / "runs",
    )
    parser.add_argument("--bfcl-data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "cof_robustness_granite300"
        / "revision_sensitivity.json",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--power-simulations", type=int, default=20_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    records = bfcl_records(args.bfcl_data_root)
    ids: list[str] | None = None
    matrices: dict[str, np.ndarray] = {}
    discordances: list[dict[str, Any]] = []

    for model, (folder, result_name) in MODELS.items():
        model_root = args.runs_root / folder
        rows = {
            policy: load_jsonl(model_root / policy / "checkpoint.jsonl")
            for policy in ("DR1", "COF1")
        }
        model_ids = [row["id"] for row in rows["DR1"]]
        if ids is None:
            ids = model_ids
        elif ids != model_ids:
            raise ValueError("Model task order differs")
        by_id = {
            policy: {row["id"]: row for row in policy_rows}
            for policy, policy_rows in rows.items()
        }
        incorrect = {
            policy: incorrect_ids(model_root, result_name, policy)
            for policy in ("DR1", "COF1")
        }
        differences = np.asarray(
            [
                int(item_id not in incorrect["COF1"])
                - int(item_id not in incorrect["DR1"])
                for item_id in model_ids
            ],
            dtype=np.int8,
        )
        matrices[model] = differences
        for item_id, difference in zip(model_ids, differences, strict=True):
            if difference == 0:
                continue
            if difference == 1:
                mechanism = (
                    f"schema_aligned_recovery_{by_id['DR1'][item_id]['initial_error_class']}"
                )
                rationale = (
                    "The constrained retry converted the validator-detectable "
                    "failure into the BFCL-accepted call while DR1 remained incorrect."
                )
            else:
                mechanism, rationale = HARMFUL_CLASSIFICATION[(model, item_id)]
            discordances.append(
                {
                    "model": model,
                    "item_id": item_id,
                    "direction": "helpful" if difference == 1 else "harmful",
                    "initial_error_class": by_id["DR1"][item_id][
                        "initial_error_class"
                    ],
                    "dr1_schema_valid": by_id["DR1"][item_id]["final_schema_valid"],
                    "cof1_schema_valid": by_id["COF1"][item_id][
                        "final_schema_valid"
                    ],
                    "dr1_output": by_id["DR1"][item_id]["result"],
                    "cof1_output": by_id["COF1"][item_id]["result"],
                    "mechanism": mechanism,
                    "classification_rationale": rationale,
                    **records[item_id],
                }
            )

    assert ids is not None
    sensitivity: dict[str, Any] = {}
    for label, model_names in {
        "two_qwen_fixed_models": ("qwen3_0.6b", "qwen3_1.7b"),
        "three_fixed_models": tuple(MODELS),
    }.items():
        matrix = np.stack([matrices[model] for model in model_names])
        sensitivity[label] = {
            "models": list(model_names),
            "estimand": (
                "mean paired BFCL-correctness difference over the frozen "
                "50:50 task mixture, conditional on the listed fixed models"
            ),
            "point_estimate_percentage_points": float(matrix.mean() * 100),
            "stratified_item_bootstrap_95_ci_percentage_points": bootstrap_ci(
                matrix,
                ids,
                rng,
                args.bootstrap_samples,
                "stratified_item",
            ),
            "context_cluster_bootstrap_95_ci_percentage_points": bootstrap_ci(
                matrix,
                ids,
                rng,
                args.bootstrap_samples,
                "context_cluster",
            ),
            "context_cluster_definition": (
                "task category plus the middle numeric component of the BFCL ID"
            ),
            "context_cluster_count": len({context_key(item_id) for item_id in ids}),
        }

    power_grid: list[dict[str, Any]] = []
    for discordance_rate in (0.02, 0.04, 0.06, 0.10):
        for difference_points in (1, 2, 3, 5):
            difference = difference_points / 100
            power_grid.append(
                {
                    "n": 300,
                    "discordance_rate": discordance_rate,
                    "paired_risk_difference_percentage_points": difference_points,
                    "simulated_exact_mcnemar_power": simulated_mcnemar_power(
                        rng,
                        300,
                        discordance_rate,
                        difference,
                        args.power_simulations,
                    ),
                }
            )

    output = {
        "analysis_type": "retrospective_revision_sensitivity",
        "not_a_preregistered_power_analysis": True,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "power_simulations": args.power_simulations,
        "sensitivity": sensitivity,
        "power_grid": power_grid,
        "discordance_count": len(discordances),
        "discordance_mechanism_counts": {
            mechanism: sum(
                row["mechanism"] == mechanism for row in discordances
            )
            for mechanism in sorted({row["mechanism"] for row in discordances})
        },
        "discordances": discordances,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in output.items() if k != "discordances"}, indent=2))


if __name__ == "__main__":
    main()
