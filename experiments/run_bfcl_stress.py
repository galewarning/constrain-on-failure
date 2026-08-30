"""Run the frozen controlled BFCL schema-error stress test."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import bfcl_eval
from openai import OpenAI

from run_bfcl_smoke import (
    feedback_for,
    format_prompt,
    load_frozen_entries,
    load_initial_records,
    load_jsonl,
    query,
    server_ready,
    wait_for_server,
)
from schemahint_agent import ValidationResult, validate_function_call


POLICIES = ("generic", "raw", "schemahint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "bfcl_stress60" / "config.json",
    )
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_one_call(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if len(matches) != 1:
        return None
    try:
        parsed = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def official_error_ids(score_dir: Path) -> set[str]:
    error_ids: set[str] = set()
    for score_path in sorted(score_dir.rglob("*_score.json")):
        rows = load_jsonl(score_path)
        error_ids.update(row["id"] for row in rows[1:])
    if not error_ids:
        raise ValueError(f"No per-item BFCL errors found under {score_dir}")
    return error_ids


def function_definition(
    entry: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any] | None:
    for item in entry["function"]:
        definition = item.get("function", item)
        if definition.get("name") == name:
            return definition
    return None


def render_call(candidate: str | Mapping[str, Any]) -> str:
    body = (
        candidate
        if isinstance(candidate, str)
        else json.dumps(candidate, ensure_ascii=False)
    )
    return f"<tool_call>\n{body}\n</tool_call>"


def wrong_type_value(type_name: str) -> Any:
    if type_name in {"string", "str"}:
        return 0
    if type_name in {"integer", "int", "float", "number"}:
        return "not-a-number"
    if type_name in {"boolean", "bool"}:
        return "not-a-boolean"
    if type_name in {"array", "list", "tuple"}:
        return {}
    if type_name in {"dict", "object"}:
        return []
    raise ValueError(f"Unsupported perturbation type: {type_name}")


def perturb(
    call: Mapping[str, Any],
    definition: Mapping[str, Any],
    error_class: str,
) -> tuple[str | dict[str, Any], str] | None:
    candidate = deepcopy(dict(call))
    arguments = candidate.get("arguments")
    if not isinstance(arguments, dict):
        return None
    schema = definition.get("parameters", {})
    properties = schema.get("properties", {})

    if error_class == "invalid_json":
        serialized = json.dumps(candidate, ensure_ascii=False)
        return serialized[:-1], "/"

    if error_class == "unknown_function":
        candidate["name"] = f"{candidate['name']}__invalid"
        return candidate, "/name"

    if error_class == "missing_required_argument":
        available = sorted(set(schema.get("required", [])) & set(arguments))
        if not available:
            return None
        target = available[0]
        del arguments[target]
        return candidate, "/"

    if error_class == "unexpected_argument":
        arguments["__unexpected_argument__"] = "invalid"
        return candidate, "/"

    if error_class == "type_mismatch":
        candidates: list[tuple[str, str]] = []
        for key in sorted(arguments):
            details = properties.get(key, {})
            type_name = details.get("type")
            if (
                isinstance(type_name, str)
                and type_name not in {"any"}
                and "enum" not in details
            ):
                try:
                    wrong_type_value(type_name)
                except ValueError:
                    continue
                candidates.append((key, type_name))
        if not candidates:
            return None
        target, type_name = candidates[0]
        arguments[target] = wrong_type_value(type_name)
        return candidate, f"/{target}"

    if error_class == "enum_mismatch":
        available = sorted(
            key
            for key in arguments
            if key in properties and "enum" in properties[key]
        )
        if not available:
            return None
        target = available[0]
        arguments[target] = "__invalid_enum__"
        return candidate, f"/{target}"

    raise ValueError(f"Unknown perturbation class: {error_class}")


def build_tasks(
    config: Mapping[str, Any],
    entries: list[dict[str, Any]],
    initial_records: Mapping[str, Mapping[str, Any]],
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    eligible: list[tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]] = []
    for entry in entries:
        item_id = entry["id"]
        if item_id in excluded_ids:
            continue
        call = extract_one_call(initial_records[item_id]["initial_result"])
        if call is None:
            continue
        definition = function_definition(entry, call.get("name"))
        if definition is None:
            continue
        validation = validate_function_call(call, entry["function"])
        if validation.valid:
            eligible.append((entry, call, definition))

    tasks: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    per_class = int(config["tasks_per_error_class"])
    for error_class in config["error_classes"]:
        selected = 0
        for entry, call, definition in eligible:
            if entry["id"] in used_ids:
                continue
            mutation = perturb(call, definition, error_class)
            if mutation is None:
                continue
            corrupted, target_path = mutation
            validation = validate_function_call(corrupted, entry["function"])
            if validation.valid or validation.hint is None:
                continue
            if validation.hint.error_class != error_class:
                continue
            tasks.append(
                {
                    "id": entry["id"],
                    "dataset": entry["_dataset"],
                    "error_class": error_class,
                    "target_path": target_path,
                    "messages": entry["question"][0],
                    "functions": entry["function"],
                    "original_call": call,
                    "corrupted_candidate": corrupted,
                    "corrupted_text": render_call(corrupted),
                    "validation": validation,
                }
            )
            used_ids.add(entry["id"])
            selected += 1
            if selected == per_class:
                break
        if selected != per_class:
            raise ValueError(
                f"Only {selected}/{per_class} tasks available for {error_class}"
            )
    return tasks


def run_policy(
    policy: str,
    tasks: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    seed: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_class: dict[str, Counter[str]] = defaultdict(Counter)

    for index, task in enumerate(tasks):
        validation: ValidationResult = task["validation"]
        feedback = feedback_for(policy, validation)
        repair_messages = [
            *[dict(message) for message in task["messages"]],
            {"role": "assistant", "content": task["corrupted_text"]},
            {"role": "user", "content": feedback},
        ]
        prompt = format_prompt(repair_messages, task["functions"])
        repair, input_tokens, output_tokens, latency = query(
            client,
            model,
            prompt,
            seed + index,
            max_tokens,
        )
        final_candidate_match = extract_one_call(repair)
        final_candidate: str | Mapping[str, Any] = (
            final_candidate_match if final_candidate_match is not None else repair
        )
        final_validation = validate_function_call(
            final_candidate,
            task["functions"],
        )
        by_class[task["error_class"]]["count"] += 1
        by_class[task["error_class"]]["schema_valid"] += int(
            final_validation.valid
        )
        records.append(
            {
                "id": task["id"],
                "_dataset": task["dataset"],
                "result": repair,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "latency": latency,
                "initial_result": task["corrupted_text"],
                "repair_result": repair,
                "repair_attempted": True,
                "initial_schema_valid": False,
                "final_schema_valid": final_validation.valid,
                "initial_error_class": task["error_class"],
                "feedback": feedback,
            }
        )

    summary = {
        "policy": policy,
        "count": len(records),
        "final_schema_valid": sum(row["final_schema_valid"] for row in records),
        "by_error_class": {
            name: dict(counts) for name, counts in sorted(by_class.items())
        },
        "total_input_tokens": sum(row["input_token_count"] for row in records),
        "total_output_tokens": sum(row["output_token_count"] for row in records),
        "total_latency_seconds": sum(row["latency"] for row in records),
    }
    return records, summary


def write_records(
    records: list[dict[str, Any]],
    output_root: Path,
) -> None:
    result_dir = (
        output_root
        / "result"
        / "Qwen_Qwen3-1.7B-FC"
        / "non_live"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({record["_dataset"] for record in records}):
        path = result_dir / f"{Path(dataset).stem}_result.json"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record["_dataset"] != dataset:
                    continue
                output = {
                    key: value for key, value in record.items() if key != "_dataset"
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source_config_path = root / config["source_config"]
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    entries = load_frozen_entries(
        source_config,
        Path(bfcl_eval.__path__[0]) / "data",
    )
    initial_records = load_initial_records(root / config["source_result_dir"])
    excluded_ids = official_error_ids(root / config["source_score_dir"])
    tasks = build_tasks(config, entries, initial_records, excluded_ids)

    run_root = args.config.resolve().parent / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            audit = {
                key: value
                for key, value in task.items()
                if key not in {"messages", "functions", "validation"}
            }
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    if args.dry_run:
        counts = Counter(task["error_class"] for task in tasks)
        print(json.dumps({"count": len(tasks), "by_error_class": counts}))
        return

    model_path = root / "models" / "Qwen3-1.7B-GGUF" / "Qwen3-1.7B-Q8_0.gguf"
    server_path = root / "tools" / "llama.cpp" / "llama-server.exe"
    model_alias = "Qwen/Qwen3-1.7B-FC"
    process: subprocess.Popen[Any] | None = None
    log_handle = None
    try:
        if not server_ready(args.port):
            log_handle = (run_root / "llama-server.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    str(server_path),
                    "-m",
                    str(model_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--alias",
                    model_alias,
                    "-t",
                    "12",
                    "-c",
                    "4096",
                    "--jinja",
                ],
                cwd=root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            wait_for_server(args.port, process)

        client = OpenAI(
            base_url=f"http://127.0.0.1:{args.port}/v1",
            api_key="none",
        )
        for policy in args.policy or POLICIES:
            records, summary = run_policy(
                policy,
                tasks,
                client,
                model_alias,
                int(config["seed"]),
                int(config["max_tokens"]),
            )
            policy_root = run_root / policy
            write_records(records, policy_root)
            (policy_root / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
