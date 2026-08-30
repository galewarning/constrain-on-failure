"""Freeze a deterministic BFCL live holdout without reading answer files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

import bfcl_eval


SPECS = (
    ("BFCL_v4_live_simple.json", 150),
    ("BFCL_v4_live_multiple.json", 150),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20260728)
    parser.add_argument("--generation-seed", type=int, default=4000)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite frozen manifest: {args.output}"
        )

    data_root = Path(bfcl_eval.__path__[0]) / "data"
    rng = random.Random(args.sample_seed)
    datasets = []
    for filename, sample_size in SPECS:
        source_path = data_root / filename
        entries = load_jsonl(source_path)
        if sample_size > len(entries):
            raise ValueError(f"{filename} has only {len(entries)} entries")
        selected_indices = sorted(rng.sample(range(len(entries)), sample_size))
        datasets.append(
            {
                "dataset": filename,
                "source_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "population_size": len(entries),
                "sample_size": sample_size,
                "sample_seed": args.sample_seed,
                "ids": [entries[index]["id"] for index in selected_indices],
            }
        )

    manifest = {
        "name": "cof_confirmatory_live300",
        "frozen_at": date.today().isoformat(),
        "selection_algorithm": (
            "Python random.Random(seed), sample indices without replacement "
            "sequentially by listed dataset, then sort indices"
        ),
        "answer_files_accessed": False,
        "datasets": datasets,
        "seed": args.generation_seed,
        "max_tokens": 512,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": sum(spec["sample_size"] for spec in datasets),
                "first": datasets[0]["ids"][0],
                "last": datasets[-1]["ids"][-1],
            }
        )
    )


if __name__ == "__main__":
    main()
