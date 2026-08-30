from pathlib import Path
import hashlib
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from run_cof_pilot import (
    extract_model_call,
    format_model_prompt,
    verify_dataset_hashes,
    write_records,
)
from prepare_granite_bfcl_results import normalize_row


def test_writes_live_and_non_live_results_to_separate_partitions(tmp_path):
    records = [
        {
            "id": "live_simple_0-0-0",
            "_dataset": "BFCL_v4_live_simple.json",
            "result": "{}",
        },
        {
            "id": "multiple_0",
            "_dataset": "BFCL_v4_multiple.json",
            "result": "{}",
        },
    ]

    write_records(records, tmp_path, "model")

    assert (
        tmp_path
        / "result"
        / "model"
        / "live"
        / "BFCL_v4_live_simple_result.json"
    ).exists()
    assert (
        tmp_path
        / "result"
        / "model"
        / "non_live"
        / "BFCL_v4_multiple_result.json"
    ).exists()


def test_verifies_frozen_dataset_hash(tmp_path):
    source = tmp_path / "sample.json"
    source.write_text('{"id":"x"}\n', encoding="utf-8")
    correct_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {
        "datasets": [
            {
                "dataset": source.name,
                "source_sha256": correct_hash,
            }
        ]
    }

    verify_dataset_hashes(config, tmp_path)
    config["datasets"][0]["source_sha256"] = "0" * 64
    try:
        verify_dataset_hashes(config, tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected an incorrect source hash to fail")


def test_extracts_single_granite_tool_call():
    text = (
        '<|tool_call|>[{"name":"weather.lookup",'
        '"arguments":{"city":"Boston"}}]</tool_call>'
    )

    assert extract_model_call(text) == {
        "name": "weather.lookup",
        "arguments": {"city": "Boston"},
    }


def test_extracts_unmarked_granite_singleton_list_but_not_trailing_text():
    text = '[{"name":"weather.lookup","arguments":{"city":"Boston"}}]'

    assert extract_model_call(text) == {
        "name": "weather.lookup",
        "arguments": {"city": "Boston"},
    }
    assert extract_model_call("Explanation\n" + text) is None


def test_granite_prompt_uses_model_native_roles_and_tools():
    prompt = format_model_prompt(
        [{"role": "user", "content": "Check Boston."}],
        [
            {
                "name": "weather.lookup",
                "description": "Look up weather.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "granite3",
    )

    assert "<|start_of_role|>tools<|end_of_role|>" in prompt
    assert "<|start_of_role|>user<|end_of_role|>Check Boston." in prompt
    assert "weather.lookup" in prompt
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_normalizes_granite_result_for_official_handler():
    row = {
        "id": "x",
        "result": '[{"name":"weather.lookup","arguments":{"city":"Boston"}}]',
    }

    assert normalize_row(row)["result"] == [
        {"weather.lookup": {"city": "Boston"}}
    ]
    assert row["result"].startswith("[")
