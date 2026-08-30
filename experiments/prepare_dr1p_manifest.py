"""Verify frozen reuse boundaries and emit the DR1-P ablation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_cof_pilot import load_jsonl
from run_cof_pilot import CONSTRAINED_RESPONSE_INSTRUCTION


MODELS = ("0.6b", "1.7b", "granite")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=base / "cof_confirmatory_live300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "cof_component_ablation_dr1p" / "manifest.json",
    )
    args = parser.parse_args()

    frozen = args.frozen_root.resolve()
    model_records: dict[str, Any] = {}
    total = 0
    for model in MODELS:
        d0_path = frozen / "runs" / model / "D0" / "checkpoint.jsonl"
        dr1_path = frozen / "runs" / model / "DR1" / "checkpoint.jsonl"
        cof1_path = frozen / "runs" / model / "COF1" / "checkpoint.jsonl"
        d0 = load_jsonl(d0_path)
        dr1 = load_jsonl(dr1_path)
        cof1 = load_jsonl(cof1_path)
        ids = [row["id"] for row in d0]
        if len(ids) != 300 or len(set(ids)) != 300:
            raise ValueError(f"{model}: D0 is not a 300-item unique manifest")
        if [row["id"] for row in dr1] != ids or [row["id"] for row in cof1] != ids:
            raise ValueError(f"{model}: policy ID order differs from D0")
        d0_by_id = {row["id"]: row for row in d0}
        invalid_ids = [row["id"] for row in d0 if not row["initial_schema_valid"]]
        dr1_triggered = [row["id"] for row in dr1 if row["repair_attempted"]]
        cof1_triggered = [row["id"] for row in cof1 if row["repair_attempted"]]
        if invalid_ids != dr1_triggered or invalid_ids != cof1_triggered:
            raise ValueError(f"{model}: frozen policy trigger sets do not match D0-invalid IDs")
        for row in (*dr1, *cof1):
            source = d0_by_id[row["id"]]
            if row["initial_result"].encode("utf-8") != source["initial_result"].encode(
                "utf-8"
            ):
                raise ValueError(f"{model}/{row['id']}: stored D0 bytes differ")
        total += len(invalid_ids)
        model_records[model] = {
            "n": len(ids),
            "d0_invalid_trigger_count": len(invalid_ids),
            "trigger_ids": invalid_ids,
            "d0_checkpoint_sha256": sha256(d0_path),
            "dr1_checkpoint_sha256": sha256(dr1_path),
            "cof1_checkpoint_sha256": sha256(cof1_path),
        }

    if total != 137:
        raise ValueError(f"Expected 137 new retries from frozen data, found {total}")
    config_path = frozen / "config.json"
    output = {
        "name": "cof_component_ablation_dr1p",
        "status": "subsequent_post_hoc_component_ablation",
        "created_at": "2026-08-30",
        "source_experiment": "cof_confirmatory_live300",
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "policy": "DR1-P",
        "policy_definition": {
            "trigger": "same deterministic D0-invalid trigger as DR1 and COF1",
            "retry_context": "same rejected call and validator diagnosis as DR1 and COF1",
            "instruction": CONSTRAINED_RESPONSE_INSTRUCTION,
            "decoder": (
                "same llama.cpp /completion endpoint and runner-applied "
                "tool-call wrapper as COF1; no json_schema field"
            ),
            "d0_valid_behavior": "return stored D0 bytes without generation",
        },
        "expected_new_generations": total,
        "models": model_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
