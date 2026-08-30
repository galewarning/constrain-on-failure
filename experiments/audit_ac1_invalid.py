"""Trace every AC1 local-validator failure from frozen live-300 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import bfcl_eval

from analyze_cof_pilot import load_jsonl
from run_cof_pilot import extract_model_call, language_for_dataset, validate_response
from schemahint_agent import build_function_call_schema


MODELS = ("0.6b", "1.7b", "granite")


def frozen_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data_root = Path(bfcl_eval.__path__[0]) / "data"
    entries: dict[str, dict[str, Any]] = {}
    for spec in config["datasets"]:
        wanted = set(spec["ids"])
        for row in load_jsonl(data_root / spec["dataset"]):
            if row["id"] in wanted:
                row["_dataset"] = spec["dataset"]
                entries[row["id"]] = row
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent / "cof_confirmatory_live300"
    parser.add_argument("--experiment-root", type=Path, default=base)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "cof_component_ablation_dr1p"
        / "ac1_invalid_audit.json",
    )
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    entries = frozen_entries(config)
    cases: list[dict[str, Any]] = []
    for model in MODELS:
        rows = load_jsonl(root / "runs" / model / "AC1" / "checkpoint.jsonl")
        for row in rows:
            if row["final_schema_valid"]:
                continue
            entry = entries[row["id"]]
            language = language_for_dataset(row["_dataset"])
            parsed = extract_model_call(row["result"])
            validation = validate_response(row["result"], entry["function"], language)
            schema = build_function_call_schema(entry["function"], language=language)
            values_schema = schema["properties"]["arguments"]["properties"].get(
                "data_values"
            )
            observed_values = (
                parsed.get("arguments", {}).get("data_values")
                if parsed is not None
                else None
            )
            number_integer_boundary = bool(
                values_schema
                and values_schema.get("items", {}).get("type") == "number"
                and isinstance(observed_values, list)
                and any(type(value) is int for value in observed_values)
                and validation.hint is not None
                and validation.hint.error_class == "type_mismatch"
            )
            cases.append(
                {
                    "model": model,
                    "bfcl_task_id": row["id"],
                    "task_category": Path(row["_dataset"]).stem,
                    "raw_generated_output": row["result"],
                    "parser_result": parsed,
                    "validator_failure_class": (
                        validation.hint.error_class if validation.hint else None
                    ),
                    "validator_path": (
                        validation.hint.path if validation.hint else None
                    ),
                    "validator_message": (
                        validation.hint.text if validation.hint else None
                    ),
                    "output_tokens": int(row["output_token_count"]),
                    "output_token_ceiling": int(config["max_tokens"]),
                    "hit_output_token_ceiling": int(row["output_token_count"])
                    >= int(config["max_tokens"]),
                    "constraint_activations": int(row["constraint_activations"]),
                    "complete_single_call_parsed": parsed is not None,
                    "generation_schema_for_data_values": values_schema,
                    "observed_data_values_python_types": (
                        [type(value).__name__ for value in observed_values]
                        if isinstance(observed_values, list)
                        else None
                    ),
                    "cause_class": (
                        "schema_compiler_expressiveness_boundary"
                        if number_integer_boundary
                        else "unresolved"
                    ),
                    "cause_detail": (
                        "Portable JSON Schema uses items:type=number, which "
                        "admits integer-valued JSON numbers, while the frozen "
                        "BFCL-aligned local validator requires exact float "
                        "instances for nested BFCL float items."
                        if number_integer_boundary
                        else None
                    ),
                    "normal_completion_evidence": {
                        "well_formed_complete_call": parsed is not None,
                        "well_below_output_ceiling": int(row["output_token_count"])
                        < int(config["max_tokens"]),
                        "one_constraint_activation_recorded": int(
                            row["constraint_activations"]
                        )
                        == 1,
                    },
                }
            )

    if len(cases) != 3:
        raise ValueError(f"Expected exactly three AC1-invalid rows, found {len(cases)}")
    if any(case["cause_class"] == "unresolved" for case in cases):
        raise ValueError("At least one AC1-invalid case remains unresolved")

    output = {
        "status": "frozen_artifact_audit",
        "case_count": len(cases),
        "unique_task_count": len({case["bfcl_task_id"] for case in cases}),
        "all_models_same_task": len({case["bfcl_task_id"] for case in cases}) == 1,
        "historical_results_invalidated": False,
        "finding": (
            "All three model rows are complete constrained outputs for the same "
            "task. They satisfy the portable generation schema but fail the "
            "stricter BFCL-aligned nested-float validator."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
