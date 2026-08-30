"""Fail-closed validation of the corrected DR1-P experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_cof_pilot import load_jsonl


MODELS = ("0.6b", "1.7b", "granite")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent
    parser.add_argument(
        "--frozen-root", type=Path, default=base / "cof_confirmatory_live300"
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=base / "cof_component_ablation_dr1p",
    )
    args = parser.parse_args()

    frozen = args.frozen_root.resolve()
    ablation = args.ablation_root.resolve()
    manifest = json.loads((ablation / "manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    total_retries = 0
    for model in MODELS:
        d0_path = frozen / "runs" / model / "D0" / "checkpoint.jsonl"
        dr1p_path = ablation / "runs" / model / "DR1-P" / "checkpoint.jsonl"
        summary_path = ablation / "runs" / model / "DR1-P" / "summary.json"
        d0 = load_jsonl(d0_path)
        dr1p = load_jsonl(dr1p_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if [row["id"] for row in dr1p] != [row["id"] for row in d0]:
            raise ValueError(f"{model}: DR1-P ID order differs from D0")
        expected = int(manifest["models"][model]["d0_invalid_trigger_count"])
        retries = sum(bool(row["repair_attempted"]) for row in dr1p)
        if retries != expected or int(summary["repairs_attempted"]) != expected:
            raise ValueError(f"{model}: expected {expected} retries, found {retries}")
        if int(summary["constraint_activations"]) != 0:
            raise ValueError(f"{model}: DR1-P activated a hard constraint")
        for source, candidate in zip(d0, dr1p, strict=True):
            if source["initial_result"].encode("utf-8") != candidate[
                "initial_result"
            ].encode("utf-8"):
                raise ValueError(f"{model}/{source['id']}: initial D0 bytes differ")
            if source["initial_schema_valid"]:
                if candidate["repair_attempted"]:
                    raise ValueError(f"{model}/{source['id']}: unexpected retry")
                if source["result"].encode("utf-8") != candidate["result"].encode(
                    "utf-8"
                ):
                    raise ValueError(f"{model}/{source['id']}: valid D0 not preserved")
            elif not candidate["repair_attempted"] or candidate["retry1_result"] is None:
                raise ValueError(f"{model}/{source['id']}: required retry missing")
        score_files = sorted(
            (ablation / "runs" / model / "DR1-P" / "score").rglob(
                "BFCL_v4_live_*_score.json"
            )
        )
        if len(score_files) != 2:
            raise ValueError(f"{model}: expected two official score files")
        total_retries += retries
        checks[model] = {
            "rows": len(dr1p),
            "retries": retries,
            "constraint_activations": int(summary["constraint_activations"]),
            "d0_valid_byte_preservation": True,
            "checkpoint_sha256": sha256(dr1p_path),
            "summary_sha256": sha256(summary_path),
            "score_sha256": {path.name: sha256(path) for path in score_files},
        }
    if total_retries != 137:
        raise ValueError(f"Expected 137 total retries, found {total_retries}")

    output = {
        "status": "pass",
        "corrected_formal_run_only": True,
        "total_new_generations": total_retries,
        "historical_conditions_regenerated": False,
        "all_dr1p_constraint_activations_zero": True,
        "all_d0_valid_outputs_byte_preserved": True,
        "models": checks,
    }
    target = ablation / "verification.json"
    target.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
