from pathlib import Path
import json
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

import run_cof_pilot as runner


FUNCTIONS = [
    {
        "name": "weather.lookup",
        "parameters": {
            "type": "dict",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def entry(item_id="item-0"):
    return {
        "id": item_id,
        "_dataset": "BFCL_v4_live_simple.json",
        "question": [[{"role": "user", "content": "Check Boston."}]],
        "function": FUNCTIONS,
    }


def direct_record(result):
    return {
        "initial_result": result,
        "input_token_count": 10,
        "output_token_count": 2,
        "latency": 0.25,
    }


def test_dr1p_preserves_valid_d0_without_generation(monkeypatch, tmp_path):
    stored = '<tool_call>\n{"name":"weather.lookup","arguments":{"city":"Boston"}}\n</tool_call>'

    def unexpected(*args, **kwargs):
        raise AssertionError("DR1-P must not generate when stored D0 is valid")

    monkeypatch.setattr(runner, "query", unexpected)
    monkeypatch.setattr(runner, "constrained_query", unexpected)
    monkeypatch.setattr(runner, "prompt_matched_unconstrained_query", unexpected)

    rows, summary = runner.run_policy(
        "DR1-P",
        [entry()],
        object(),
        "model",
        1053,
        4000,
        512,
        "qwen",
        {"item-0": direct_record(stored)},
        None,
        tmp_path / "checkpoint.jsonl",
    )

    assert rows[0]["result"] == stored
    assert rows[0]["initial_result"] == stored
    assert rows[0]["repair_attempted"] is False
    assert rows[0]["constraint_activations"] == 0
    assert summary["repairs_attempted"] == 0
    assert (tmp_path / "checkpoint.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_dr1p_matches_cof1_context_and_instruction_but_is_unconstrained(
    monkeypatch, tmp_path
):
    stored = "not a function call"
    captured = {}

    def fake_prompt(messages, functions, prompt_family):
        captured["messages"] = messages
        captured["functions"] = functions
        captured["prompt_family"] = prompt_family
        return "matched-prompt"

    def fake_query(port, prompt, seed, max_tokens):
        captured["query"] = (port, prompt, seed, max_tokens)
        return (
            '<tool_call>\n{"name":"weather.lookup","arguments":{"city":"Boston"}}\n</tool_call>',
            20,
            8,
            0.5,
        )

    def forbidden_constraint(*args, **kwargs):
        raise AssertionError("DR1-P must not pass JSON Schema to llama.cpp")

    monkeypatch.setattr(runner, "prompt_matched_retry_prompt", fake_prompt)
    monkeypatch.setattr(runner, "prompt_matched_unconstrained_query", fake_query)
    monkeypatch.setattr(runner, "constrained_query", forbidden_constraint)

    rows, summary = runner.run_policy(
        "DR1-P",
        [entry()],
        object(),
        "model",
        1053,
        4000,
        512,
        "granite3",
        {"item-0": direct_record(stored)},
        None,
        tmp_path / "checkpoint.jsonl",
    )

    assert captured["messages"] == runner.diagnostic_messages(
        entry()["question"][0],
        stored,
        runner.validate_response(stored, FUNCTIONS, "python"),
    )
    assert captured["functions"] == FUNCTIONS
    assert captured["prompt_family"] == "granite3"
    assert captured["query"] == (1053, "matched-prompt", 4000, 512)
    assert rows[0]["repair_attempted"] is True
    assert rows[0]["constraint_activations"] == 0
    assert rows[0]["final_schema_valid"] is True
    assert summary["repairs_attempted"] == 1
    assert summary["constraint_activations"] == 0


def test_dr1p_uses_cof1_instruction_verbatim():
    messages = [{"role": "user", "content": "Check Boston."}]

    assert runner.prompt_matched_retry_prompt(
        messages, FUNCTIONS, "qwen"
    ) == runner.constrained_prompt(messages, FUNCTIONS, "qwen")


def test_dr1p_completion_request_omits_json_schema_and_matches_wrapper(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {
                    "content": '{"name":"weather.lookup","arguments":{"city":"Boston"}}',
                    "tokens_evaluated": 12,
                    "tokens_predicted": 5,
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
    text, input_tokens, output_tokens, _ = runner.prompt_matched_unconstrained_query(
        1053, "prompt", 4000, 512
    )

    assert captured["url"].endswith("/completion")
    assert "json_schema" not in captured["payload"]
    assert captured["payload"] == {
        "prompt": "prompt",
        "n_predict": 512,
        "temperature": 0,
        "seed": 4000,
        "cache_prompt": True,
    }
    assert text.startswith("<tool_call>\n") and text.endswith("\n</tool_call>")
    assert (input_tokens, output_tokens) == (12, 5)
