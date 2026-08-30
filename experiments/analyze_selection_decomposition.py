"""Decompose always-constrained versus selective recovery by D0 validity.

This analysis uses only retained generation checkpoints and official BFCL
score files. It does not run a model or modify frozen experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = ("live_simple", "live_multiple")
POLICIES = ("D0", "AC1", "DR1", "COF1")
RESULT_NAMES = {
    "qwen3_0.6b": ("0.6b", "Qwen_Qwen3-0.6B-FC"),
    "qwen3_1.7b": ("1.7b", "Qwen_Qwen3-1.7B-FC"),
    "granite_3.3_2b": (
        "granite",
        "ibm-granite_granite-3.2-8b-instruct",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def incorrect_ids(policy_root: Path, result_name: str) -> set[str]:
    score_root = policy_root / "score" / result_name / "live"
    incorrect: set[str] = set()
    for category in CATEGORIES:
        rows = load_jsonl(score_root / f"BFCL_v4_{category}_score.json")
        incorrect.update(row["id"] for row in rows[1:])
    return incorrect


def paired_transition(
    candidate: dict[str, bool],
    baseline: dict[str, bool],
    ids: list[str],
) -> dict[str, int]:
    helpful = sum(not baseline[item_id] and candidate[item_id] for item_id in ids)
    harmful = sum(baseline[item_id] and not candidate[item_id] for item_id in ids)
    return {
        "candidate_helpful": helpful,
        "candidate_harmful": harmful,
        "correctness_ties": len(ids) - helpful - harmful,
        "net_correct": helpful - harmful,
    }


def context_key(item_id: str) -> str:
    category, triplet = item_id.rsplit("_", 1)
    return f"{category}:{triplet.split('-')[1]}"


def analyze_model(model_root: Path, result_name: str) -> dict[str, Any]:
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
        policy: {row["id"]: row for row in policy_rows}
        for policy, policy_rows in rows.items()
    }
    incorrect = {
        policy: incorrect_ids(model_root / policy, result_name)
        for policy in POLICIES
    }
    correct = {
        policy: {item_id: item_id not in incorrect[policy] for item_id in ids}
        for policy in POLICIES
    }
    strata = {
        "D0_valid": [
            item_id
            for item_id in ids
            if bool(by_id["D0"][item_id]["initial_schema_valid"])
        ],
        "D0_invalid": [
            item_id
            for item_id in ids
            if not bool(by_id["D0"][item_id]["initial_schema_valid"])
        ],
    }

    output: dict[str, Any] = {"n": len(ids), "strata": {}}
    for stratum, stratum_ids in strata.items():
        output["strata"][stratum] = {
            "n": len(stratum_ids),
            "correct": {
                policy: sum(correct[policy][item_id] for item_id in stratum_ids)
                for policy in POLICIES
            },
            "COF1_vs_AC1": paired_transition(
                correct["COF1"], correct["AC1"], stratum_ids
            ),
            "COF1_vs_D0": paired_transition(
                correct["COF1"], correct["D0"], stratum_ids
            ),
        }

    valid_ids = strata["D0_valid"]
    output["D0_valid_preservation_checks"] = {
        "cof1_final_equals_d0_final_count": sum(
            by_id["COF1"][item_id]["result"] == by_id["D0"][item_id]["result"]
            for item_id in valid_ids
        ),
        "cof1_correctness_equals_d0_count": sum(
            correct["COF1"][item_id] == correct["D0"][item_id]
            for item_id in valid_ids
        ),
        "cof1_retry_count": sum(
            bool(by_id["COF1"][item_id]["repair_attempted"])
            for item_id in valid_ids
        ),
    }
    output["overall_COF1_vs_AC1"] = paired_transition(
        correct["COF1"], correct["AC1"], ids
    )
    return output


def combined_counts(
    per_model: dict[str, dict[str, Any]],
    model_names: list[str],
) -> dict[str, Any]:
    combined: dict[str, Any] = {"models": model_names, "strata": {}}
    for stratum in ("D0_valid", "D0_invalid"):
        combined["strata"][stratum] = {
            "n": sum(per_model[model]["strata"][stratum]["n"] for model in model_names),
            "correct": {
                policy: sum(
                    per_model[model]["strata"][stratum]["correct"][policy]
                    for model in model_names
                )
                for policy in POLICIES
            },
            "COF1_vs_AC1": {
                field: sum(
                    per_model[model]["strata"][stratum]["COF1_vs_AC1"][field]
                    for model in model_names
                )
                for field in (
                    "candidate_helpful",
                    "candidate_harmful",
                    "correctness_ties",
                    "net_correct",
                )
            },
        }
    return combined


def cluster_distribution(ids: list[str]) -> dict[str, Any]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        clusters[context_key(item_id)].append(item_id)
    sizes = [len(items) for items in clusters.values()]
    counts = Counter(sizes)
    return {
        "construction": "task category plus middle numeric BFCL-ID component",
        "interpretation": (
            "heuristic dependence sensitivity only; no benchmark documentation "
            "was used to establish this ID component as a semantic context unit"
        ),
        "cluster_count": len(sizes),
        "minimum_size": min(sizes),
        "median_size": sorted(sizes)[len(sizes) // 2],
        "maximum_size": max(sizes),
        "singleton_count": counts[1],
        "size_frequency": {str(size): counts[size] for size in sorted(counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).parent / "cof_confirmatory_live300" / "runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_model = {
        model: analyze_model(args.runs_root / folder, result_name)
        for model, (folder, result_name) in RESULT_NAMES.items()
    }
    first_rows = load_jsonl(
        args.runs_root / "0.6b" / "D0" / "checkpoint.jsonl"
    )
    output = {
        "analysis_type": "stored_output_selection_decomposition",
        "model_generation_performed": False,
        "models": per_model,
        "combined": {
            "two_qwen_fixed_instances": combined_counts(
                per_model, ["qwen3_0.6b", "qwen3_1.7b"]
            ),
            "three_fixed_instances": combined_counts(
                per_model, list(RESULT_NAMES)
            ),
        },
        "context_cluster_sensitivity": cluster_distribution(
            [row["id"] for row in first_rows]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
