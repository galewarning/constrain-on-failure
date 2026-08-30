"""Run frozen Constrain-on-Failure policies on BFCL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import bfcl_eval
from bfcl_eval.model_handler.local_inference.granite_3 import Granite3FCHandler
from openai import OpenAI

from run_bfcl_smoke import (
    extract_candidate,
    feedback_for,
    format_prompt as format_qwen_prompt,
    load_frozen_entries,
    load_initial_records,
    load_jsonl,
    query,
    server_ready,
    wait_for_server,
)
from run_bfcl_stress import extract_one_call
from schemahint_agent import (
    ValidationResult,
    build_function_call_schema,
    validate_function_call,
)


POLICIES = ("D0", "AC1", "DR1", "DR1-P", "COF1", "DR2", "GCOF2")
CONSTRAINED_RESPONSE_INSTRUCTION = (
    "For this constrained response, return only one JSON object with "
    "exactly the keys name and arguments. Preserve the user's requested "
    "values. Do not use XML tags or explanatory text."
)
MODELS = {
    "0.6b": {
        "path": "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        "alias": "Qwen/Qwen3-0.6B-FC",
        "result_name": "Qwen_Qwen3-0.6B-FC",
        "prompt_family": "qwen",
    },
    "1.7b": {
        "path": "models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf",
        "alias": "Qwen/Qwen3-1.7B-FC",
        "result_name": "Qwen_Qwen3-1.7B-FC",
        "prompt_family": "qwen",
    },
    "granite": {
        "path": (
            "models/granite-3.3-2b-instruct-GGUF/"
            "granite-3.3-2b-instruct-Q8_0.gguf"
        ),
        "alias": "ibm-granite/granite-3.3-2b-instruct-FC",
        "result_name": "ibm-granite_granite-3.3-2b-instruct-FC",
        "prompt_family": "granite3",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "cof_pilot228" / "config.json",
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--direct-result-dir", type=Path)
    parser.add_argument("--dr1-result-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional experiment output directory. Defaults to the directory "
            "containing --config. Use this for subsequent analyses so frozen "
            "experiment artifacts are never overwritten."
        ),
    )
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_response(
    text: str,
    functions: list[dict[str, Any]],
    language: str,
) -> ValidationResult:
    candidate = extract_model_call(text)
    return validate_function_call(
        candidate if candidate is not None else extract_candidate(text),
        functions,
        language=language,
    )


def extract_model_call(text: str) -> dict[str, Any] | None:
    """Extract one Qwen- or Granite-style function call for local validation."""

    candidate = extract_one_call(text)
    if candidate is not None:
        return candidate

    # Granite 3.3 commonly emits its documented JSON list of tool calls
    # without repeating the optional textual marker. Accept only a complete
    # singleton list with the exact call shape; explanatory or trailing text
    # remains invalid.
    try:
        plain = json.loads(text.strip())
    except json.JSONDecodeError:
        plain = None
    if (
        isinstance(plain, list)
        and len(plain) == 1
        and isinstance(plain[0], dict)
        and set(plain[0]) == {"name", "arguments"}
        and isinstance(plain[0]["name"], str)
        and isinstance(plain[0]["arguments"], dict)
    ):
        return plain[0]

    # BFCL's Granite helper recognizes both opening markers but removes the
    # optional closing tag only for the non-pipe spelling. Normalize it first
    # so either valid Granite rendering is handled identically.
    extracted = Granite3FCHandler._extract_tool_calls(
        text.replace("</tool_call>", "")
    )
    if (
        len(extracted) == 1
        and isinstance(extracted[0], dict)
        and isinstance(extracted[0].get("name"), str)
        and isinstance(extracted[0].get("arguments"), dict)
    ):
        return extracted[0]
    return None


def format_model_prompt(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    prompt_family: str,
) -> str:
    """Apply the frozen BFCL model-family formatter."""

    if prompt_family == "qwen":
        return format_qwen_prompt(messages, functions)
    if prompt_family == "granite3":
        prompt = Granite3FCHandler._format_prompt(None, messages, functions)
        # The prose Jinja template embedded in BFCL's Granite handler specifies
        # an assistant generation prompt, but its manual formatter omits that
        # final role marker. Add the model-native marker here so completion
        # generation begins inside the assistant turn.
        return prompt + "<|start_of_role|>assistant<|end_of_role|>"
    raise ValueError(f"Unsupported prompt family: {prompt_family}")


def language_for_dataset(dataset: str) -> str:
    if "simple_java.json" in dataset:
        return "java"
    if "simple_javascript.json" in dataset:
        return "javascript"
    return "python"


def verify_dataset_hashes(
    config: Mapping[str, Any],
    data_root: Path,
) -> None:
    for spec in config.get("datasets", []):
        expected = spec.get("source_sha256")
        if expected is None:
            continue
        data_path = data_root / spec["dataset"]
        actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Dataset hash mismatch for {data_path.name}: "
                f"expected {expected}, got {actual}"
            )


def render_call(content: str) -> str:
    return f"<tool_call>\n{content.strip()}\n</tool_call>"


def constrained_query(
    port: int,
    prompt: str,
    schema: dict[str, Any],
    seed: int,
    max_tokens: int,
) -> tuple[str, int, int, float]:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "seed": seed,
        "json_schema": schema,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    latency = time.perf_counter() - started
    return (
        render_call(result["content"]),
        int(result.get("tokens_evaluated", 0)),
        int(result.get("tokens_predicted", 0)),
        latency,
    )


def prompt_matched_unconstrained_query(
    port: int,
    prompt: str,
    seed: int,
    max_tokens: int,
) -> tuple[str, int, int, float]:
    """Match COF1's request path and post-processing without json_schema."""

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "seed": seed,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    latency = time.perf_counter() - started
    return (
        render_call(result["content"]),
        int(result.get("tokens_evaluated", 0)),
        int(result.get("tokens_predicted", 0)),
        latency,
    )


def constrained_prompt(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    prompt_family: str,
) -> str:
    override = {
        "role": "user",
        "content": CONSTRAINED_RESPONSE_INSTRUCTION,
    }
    return format_model_prompt([*messages, override], functions, prompt_family)


def prompt_matched_retry_prompt(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    prompt_family: str,
) -> str:
    """Use COF1's exact retry wording without enabling decoder constraints."""

    return constrained_prompt(messages, functions, prompt_family)


def diagnostic_messages(
    messages: list[dict[str, Any]],
    failed_text: str,
    validation: ValidationResult,
) -> list[dict[str, Any]]:
    return [
        *messages,
        {"role": "assistant", "content": failed_text},
        {"role": "user", "content": feedback_for("schemahint", validation)},
    ]


def record_for(
    entry: Mapping[str, Any],
    result: str,
    initial: str,
    retry1: str | None,
    retry2: str | None,
    initial_validation: ValidationResult,
    final_validation: ValidationResult,
    input_tokens: int,
    output_tokens: int,
    latency: float,
    constraint_activations: int,
) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "_dataset": entry["_dataset"],
        "result": result,
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "latency": latency,
        "initial_result": initial,
        "retry1_result": retry1,
        "retry2_result": retry2,
        "repair_attempted": retry1 is not None,
        "second_repair_attempted": retry2 is not None,
        "constraint_activations": constraint_activations,
        "initial_schema_valid": initial_validation.valid,
        "final_schema_valid": final_validation.valid,
        "initial_error_class": (
            initial_validation.hint.error_class
            if initial_validation.hint is not None
            else None
        ),
    }


def run_policy(
    policy: str,
    entries: list[dict[str, Any]],
    client: OpenAI,
    model_alias: str,
    port: int,
    seed: int,
    max_tokens: int,
    prompt_family: str,
    direct_records: dict[str, dict[str, Any]] | None,
    dr1_records: dict[str, dict[str, Any]] | None,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    error_classes: Counter[str] = Counter()
    checkpoint_records = (
        {row["id"]: row for row in load_jsonl(checkpoint_path)}
        if checkpoint_path.exists()
        else {}
    )

    for index, entry in enumerate(entries):
        if entry["id"] in checkpoint_records:
            row = checkpoint_records[entry["id"]]
            records.append(row)
            if row.get("initial_error_class") is not None:
                error_classes[row["initial_error_class"]] += 1
            continue
        messages = [dict(message) for message in entry["question"][0]]
        functions = entry["function"]
        language = language_for_dataset(entry["_dataset"])
        item_seed = seed + index
        retry1: str | None = None
        retry2: str | None = None
        constraint_activations = 0

        if policy == "AC1":
            prompt = constrained_prompt(messages, functions, prompt_family)
            initial, input_tokens, output_tokens, latency = constrained_query(
                port,
                prompt,
                build_function_call_schema(functions, language=language),
                item_seed,
                max_tokens,
            )
            constraint_activations = 1
        elif policy == "D0":
            if direct_records is not None:
                source = direct_records[entry["id"]]
                initial = source["initial_result"]
                input_tokens = int(source["input_token_count"])
                output_tokens = int(source["output_token_count"])
                latency = float(source["latency"])
            else:
                prompt = format_model_prompt(messages, functions, prompt_family)
                initial, input_tokens, output_tokens, latency = query(
                    client,
                    model_alias,
                    prompt,
                    item_seed,
                    max_tokens,
                )
        else:
            assert direct_records is not None
            source = direct_records[entry["id"]]
            initial = source["initial_result"]
            input_tokens = int(source["input_token_count"])
            output_tokens = int(source["output_token_count"])
            latency = float(source["latency"])

        initial_validation = validate_response(initial, functions, language)
        if initial_validation.hint is not None:
            error_classes[initial_validation.hint.error_class] += 1
        final = initial

        if policy in {"DR1", "DR1-P", "DR2", "GCOF2"} and not initial_validation.valid:
            if policy in {"DR2", "GCOF2"} and dr1_records is not None:
                first_source = dr1_records[entry["id"]]
                retry1 = first_source.get("retry1_result")
                final = first_source["result"]
                input_tokens = int(first_source["input_token_count"])
                output_tokens = int(first_source["output_token_count"])
                latency = float(first_source["latency"])
            else:
                repair_messages = diagnostic_messages(
                    messages,
                    initial,
                    initial_validation,
                )
                if policy == "DR1-P":
                    prompt = prompt_matched_retry_prompt(
                        repair_messages,
                        functions,
                        prompt_family,
                    )
                else:
                    prompt = format_model_prompt(
                        repair_messages,
                        functions,
                        prompt_family,
                    )
                if policy == "DR1-P":
                    retry1, added_in, added_out, added_latency = (
                        prompt_matched_unconstrained_query(
                            port,
                            prompt,
                            item_seed,
                            max_tokens,
                        )
                    )
                else:
                    retry1, added_in, added_out, added_latency = query(
                        client,
                        model_alias,
                        prompt,
                        item_seed,
                        max_tokens,
                    )
                final = retry1
                input_tokens += added_in
                output_tokens += added_out
                latency += added_latency

        elif policy == "COF1" and not initial_validation.valid:
            repair_messages = diagnostic_messages(
                messages,
                initial,
                initial_validation,
            )
            prompt = constrained_prompt(
                repair_messages,
                functions,
                prompt_family,
            )
            retry1, added_in, added_out, added_latency = constrained_query(
                port,
                prompt,
                build_function_call_schema(functions, language=language),
                item_seed,
                max_tokens,
            )
            constraint_activations = 1
            final = retry1
            input_tokens += added_in
            output_tokens += added_out
            latency += added_latency

        first_final_validation = validate_response(final, functions, language)
        if (
            policy in {"DR2", "GCOF2"}
            and retry1 is not None
            and not first_final_validation.valid
        ):
            second_messages = diagnostic_messages(
                diagnostic_messages(messages, initial, initial_validation),
                final,
                first_final_validation,
            )
            if policy == "DR2":
                prompt = format_model_prompt(
                    second_messages,
                    functions,
                    prompt_family,
                )
                retry2, added_in, added_out, added_latency = query(
                    client,
                    model_alias,
                    prompt,
                    item_seed,
                    max_tokens,
                )
            else:
                prompt = constrained_prompt(
                    second_messages,
                    functions,
                    prompt_family,
                )
                retry2, added_in, added_out, added_latency = constrained_query(
                    port,
                    prompt,
                    build_function_call_schema(functions, language=language),
                    item_seed,
                    max_tokens,
                )
                constraint_activations = 1
            final = retry2
            input_tokens += added_in
            output_tokens += added_out
            latency += added_latency

        final_validation = validate_response(final, functions, language)
        row = record_for(
            entry,
            final,
            initial,
            retry1,
            retry2,
            initial_validation,
            final_validation,
            input_tokens,
            output_tokens,
            latency,
            constraint_activations,
        )
        records.append(row)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "policy": policy,
        "count": len(records),
        "initial_schema_valid": sum(row["initial_schema_valid"] for row in records),
        "final_schema_valid": sum(row["final_schema_valid"] for row in records),
        "repairs_attempted": sum(row["repair_attempted"] for row in records),
        "second_repairs_attempted": sum(
            row["second_repair_attempted"] for row in records
        ),
        "constraint_activations": sum(
            row["constraint_activations"] for row in records
        ),
        "initial_error_classes": dict(sorted(error_classes.items())),
        "total_input_tokens": sum(row["input_token_count"] for row in records),
        "total_output_tokens": sum(row["output_token_count"] for row in records),
        "total_latency_seconds": sum(row["latency"] for row in records),
    }
    return records, summary


def write_records(
    records: list[dict[str, Any]],
    policy_root: Path,
    result_name: str,
) -> None:
    for dataset in sorted({record["_dataset"] for record in records}):
        partition = "live" if "_live_" in dataset else "non_live"
        result_dir = policy_root / "result" / result_name / partition
        result_dir.mkdir(parents=True, exist_ok=True)
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
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(bfcl_eval.__path__[0]) / "data"
    verify_dataset_hashes(config, data_root)
    entries = load_frozen_entries(
        config,
        data_root,
    )
    if args.limit is not None:
        entries = entries[: args.limit]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "count": len(entries),
                    "first": entries[0]["id"],
                    "last": entries[-1]["id"],
                }
            )
        )
        return

    model = MODELS[args.model]
    model_path = root / model["path"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    direct_records = (
        load_initial_records(args.direct_result_dir.resolve())
        if args.direct_result_dir is not None
        else None
    )
    dr1_records = (
        load_initial_records(args.dr1_result_dir.resolve())
        if args.dr1_result_dir is not None
        else None
    )
    if args.policy not in {"D0", "AC1"} and direct_records is None:
        raise ValueError(f"{args.policy} requires --direct-result-dir")
    if args.policy in {"DR2", "GCOF2"} and dr1_records is None:
        raise ValueError(f"{args.policy} requires --dr1-result-dir")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else config_path.parent
    )
    run_root = output_root / "runs" / args.model / args.policy
    run_root.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[Any] | None = None
    log_handle = None
    try:
        if not server_ready(args.port):
            log_handle = (run_root / "llama-server.log").open(
                "w",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [
                    str(root / "tools" / "llama.cpp" / "llama-server.exe"),
                    "-m",
                    str(model_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--alias",
                    model["alias"],
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
        records, summary = run_policy(
            args.policy,
            entries,
            client,
            model["alias"],
            args.port,
            int(config["seed"]),
            int(config["max_tokens"]),
            model["prompt_family"],
            direct_records,
            dr1_records,
            run_root / "checkpoint.jsonl",
        )
        write_records(records, run_root, model["result_name"])
        (run_root / "summary.json").write_text(
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
