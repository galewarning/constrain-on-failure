"""Prepare immutable Granite runner outputs for the official BFCL scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_cof_pilot import extract_model_call


SCORER_ALIAS = "ibm-granite_granite-3.2-8b-instruct"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a Granite call to the decoded AST shape required by BFCL."""

    output = dict(row)
    raw = row["result"]
    if isinstance(raw, str):
        call = extract_model_call(raw)
        if call is not None:
            output["result"] = [{call["name"]: call["arguments"]}]
    return output


def prepare(input_root: Path, output_root: Path) -> dict[str, int]:
    files = sorted(input_root.glob("*_result.json"))
    if not files:
        raise FileNotFoundError(f"No BFCL result files under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    normalized = 0
    for source in files:
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prepared: list[dict[str, Any]] = []
        for row in rows:
            converted = normalize_row(row)
            normalized += int(isinstance(converted["result"], list))
            total += 1
            prepared.append(converted)
        target = output_root / source.name
        target.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in prepared
            ),
            encoding="utf-8",
        )
    return {"total": total, "normalized": normalized}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.input_root.resolve(), args.output_root.resolve()),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
