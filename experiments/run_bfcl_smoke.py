"""Run a frozen BFCL config through controlled repair policies."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import bfcl_eval
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from openai import OpenAI

from schemahint_agent import ValidationResult, validate_function_call


POLICIES = ("direct", "generic", "raw", "schemahint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        help="Frozen experiment config; defaults to the original smoke config.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Output directory; defaults to a runs directory beside the config.",
    )
    parser.add_argument(
        "--reuse-initial-from",
        type=Path,
        help=(
            "Directory containing prior BFCL result JSON files. Reuse their "
            "frozen initial outputs and generate only policy retries."
        ),
    )
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expand_dataset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand explicit IDs or inclusive-start/exclusive-stop frozen ranges."""

    if "datasets" not in config:
        return [{"dataset": config["dataset"], "ids": config["ids"]}]

    expanded: list[dict[str, Any]] = []
    for spec in config["datasets"]:
        if "ids" in spec:
            ids = list(spec["ids"])
        else:
            ids = [
                f"{spec['id_prefix']}_{index}"
                for index in range(spec["start"], spec["stop"])
            ]
        expanded.append({"dataset": spec["dataset"], "ids": ids})
    return expanded


def load_frozen_entries(
    config: dict[str, Any],
    data_root: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in expand_dataset_specs(config):
        data_path = data_root / spec["dataset"]
        available = {entry["id"]: entry for entry in load_jsonl(data_path)}
        missing = [item_id for item_id in spec["ids"] if item_id not in available]
        if missing:
            raise KeyError(f"Missing BFCL IDs in {data_path.name}: {missing}")
        for item_id in spec["ids"]:
            entry = dict(available[item_id])
            entry["_dataset"] = spec["dataset"]
            entries.append(entry)
    return entries


def load_initial_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for result_path in sorted(path.rglob("*_result.json")):
        for record in load_jsonl(result_path):
            item_id = record["id"]
            if item_id in records:
                raise ValueError(f"Duplicate reused initial record: {item_id}")
            records[item_id] = record
    if not records:
        raise ValueError(f"No BFCL result files found under {path}")
    return records


def clean_reasoning(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def extract_candidate(text: str) -> str | dict[str, Any]:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not matches:
        return text
    try:
        parsed = json.loads(matches[0])
    except json.JSONDecodeError:
        return matches[0]
    return parsed


def feedback_for(policy: str, validation: ValidationResult) -> str:
    prefix = (
        "Your previous function call was invalid. Make exactly one corrected "
        "function call using the supplied tools. "
    )
    if policy == "raw":
        return prefix + f"Validator error: {validation.raw_error}"
    if policy == "generic":
        return prefix
    if policy == "schemahint":
        assert validation.hint is not None
        return prefix + f"Schema-derived correction: {validation.hint.text}"
    raise ValueError(f"No retry feedback for policy {policy}")


def query(
    client: OpenAI,
    model: str,
    prompt: str,
    seed: int,
    max_tokens: int,
) -> tuple[str, int, int, float]:
    started = time.perf_counter()
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        seed=seed,
    )
    latency = time.perf_counter() - started
    usage = response.usage
    return (
        clean_reasoning(response.choices[0].text),
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        latency,
    )


def format_prompt(messages: list[dict[str, Any]], functions: list[dict[str, Any]]) -> str:
    # BFCL is pinned, and this deliberately reuses its Qwen FC prompt formatter.
    prompt = QwenFCHandler._format_prompt(None, messages, functions)
    # Qwen's own template emits this prefix when enable_thinking=false.
    # It is held constant across policies to keep CPU cost bounded.
    return prompt + "<think>\n\n</think>\n\n"


def run_policy(
    policy: str,
    entries: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    seed: int,
    max_tokens: int,
    initial_records: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    error_classes: Counter[str] = Counter()

    for index, entry in enumerate(entries):
        messages = [dict(message) for message in entry["question"][0]]
        functions = entry["function"]
        prompt = format_prompt(messages, functions)
        reused = initial_records.get(entry["id"]) if initial_records else None
        if reused is None:
            initial, input_tokens, output_tokens, latency = query(
                client, model, prompt, seed + index, max_tokens
            )
        else:
            initial = reused.get("initial_result", reused["result"])
            input_tokens = reused["input_token_count"]
            output_tokens = reused["output_token_count"]
            latency = reused["latency"]
        validation = validate_function_call(extract_candidate(initial), functions)
        final = initial
        repair: str | None = None
        feedback: str | None = None

        if not validation.valid:
            assert validation.hint is not None
            error_classes[validation.hint.error_class] += 1

        if policy != "direct" and not validation.valid:
            feedback = feedback_for(policy, validation)
            repair_messages = messages + [
                {"role": "assistant", "content": initial},
                {"role": "user", "content": feedback},
            ]
            repair_prompt = format_prompt(repair_messages, functions)
            repair, repair_input, repair_output, repair_latency = query(
                client, model, repair_prompt, seed + index, max_tokens
            )
            final = repair
            input_tokens += repair_input
            output_tokens += repair_output
            latency += repair_latency

        final_validation = validate_function_call(extract_candidate(final), functions)
        records.append(
            {
                "id": entry["id"],
                "_dataset": entry["_dataset"],
                "result": final,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "latency": latency,
                "initial_result": initial,
                "repair_result": repair,
                "repair_attempted": repair is not None,
                "initial_schema_valid": validation.valid,
                "final_schema_valid": final_validation.valid,
                "initial_error_class": (
                    validation.hint.error_class if validation.hint else None
                ),
                "feedback": feedback,
            }
        )

    summary = {
        "policy": policy,
        "count": len(records),
        "initial_schema_valid": sum(row["initial_schema_valid"] for row in records),
        "final_schema_valid": sum(row["final_schema_valid"] for row in records),
        "repairs_attempted": sum(row["repair_attempted"] for row in records),
        "initial_error_classes": dict(sorted(error_classes.items())),
        "total_input_tokens": sum(row["input_token_count"] for row in records),
        "total_output_tokens": sum(row["output_token_count"] for row in records),
        "total_latency_seconds": sum(row["latency"] for row in records),
    }
    return records, summary


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def server_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(health_url(port), timeout=1) as response:
            payload = json.loads(response.read())
        return payload.get("status") == "ok"
    except Exception:
        return False


def wait_for_server(port: int, process: subprocess.Popen[Any], timeout: int = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        if server_ready(port):
            return
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else root / "experiments" / "bfcl_smoke20" / "test_case_ids.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = load_frozen_entries(
        config,
        Path(bfcl_eval.__path__[0]) / "data",
    )
    if args.limit is not None:
        entries = entries[: args.limit]
    initial_records = (
        load_initial_records(args.reuse_initial_from.resolve())
        if args.reuse_initial_from is not None
        else None
    )
    if initial_records is not None:
        missing_initials = [
            entry["id"] for entry in entries if entry["id"] not in initial_records
        ]
        if missing_initials:
            raise KeyError(f"Missing reused initial outputs: {missing_initials}")

    model_path = root / "models" / "Qwen3-1.7B-GGUF" / "Qwen3-1.7B-Q8_0.gguf"
    server_path = root / "tools" / "llama.cpp" / "llama-server.exe"
    model_alias = "Qwen/Qwen3-1.7B-FC"
    run_root = (
        args.run_root.resolve()
        if args.run_root is not None
        else config_path.parent / "runs"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    owns_server = not server_ready(args.port)
    process: subprocess.Popen[Any] | None = None
    log_handle = None
    try:
        if owns_server:
            log_handle = (run_root / "llama-server.log").open("w", encoding="utf-8")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
                creationflags=creationflags,
            )
            wait_for_server(args.port, process)

        client = OpenAI(
            base_url=f"http://127.0.0.1:{args.port}/v1",
            api_key="none",
        )
        policies = args.policy or list(POLICIES)
        for policy in policies:
            records, summary = run_policy(
                policy,
                entries,
                client,
                model_alias,
                args.seed,
                args.max_tokens,
                initial_records,
            )
            policy_dir = run_root / policy
            result_dir = (
                policy_dir
                / "result"
                / "Qwen_Qwen3-1.7B-FC"
                / "non_live"
            )
            result_dir.mkdir(parents=True, exist_ok=True)
            datasets = sorted({record["_dataset"] for record in records})
            for dataset in datasets:
                result_path = result_dir / f"{Path(dataset).stem}_result.json"
                with result_path.open("w", encoding="utf-8") as handle:
                    for record in records:
                        if record["_dataset"] != dataset:
                            continue
                        output_record = {
                            key: value
                            for key, value in record.items()
                            if key != "_dataset"
                        }
                        handle.write(
                            json.dumps(output_record, ensure_ascii=False) + "\n"
                        )
            (policy_dir / "summary.json").write_text(
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
