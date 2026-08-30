# Granite Robustness Extension Protocol

Frozen: 2026-07-27

Compatibility status: passed on 2026-07-27; see
`COMPATIBILITY_RESULTS.md`.

## Status

The user confirmed the Stage 1 addendum design before model download or
Granite generation. Draft v1 and the original two-Qwen confirmation remain
unchanged.

## Model

- `ibm-granite/granite-3.3-2b-instruct-GGUF`
- `granite-3.3-2b-instruct-Q8_0.gguf`
- CPU-only llama.cpp `b10141`
- 12 threads, 4,096-token context, temperature zero
- repository revision:
  `7cdf86ccd1f1bb3491c9b7017b033f2e51367397`
- file size: `2,694,120,096` bytes
- SHA-256:
  `39068a5bcb15f10487a49426d617b1b0f38a624d7dbdefe5ff6f26de82d97039`

## Prompt and parsing

Unconstrained generations use the `Granite3FCHandler` formatter and
Granite-style tool-call extraction included in the frozen
`bfcl-eval==2026.3.23` installation. The Qwen-specific formatter is prohibited
for this model.

The installed handler's embedded Jinja specification includes an assistant
generation prompt, while its manual Python formatter omits the final assistant
role marker. The adapter restores
`<|start_of_role|>assistant<|end_of_role|>` after the otherwise unchanged BFCL
prompt. This model-interface correction was identified by the compatibility
gate before any live-300 Granite generation.

Granite 3.3 also emits its native JSON list of tool calls without always
repeating the optional `<|tool_call|>` text marker. The adapter accepts an
unmarked response only when the entire response parses as a singleton JSON
list containing exactly `name` and object-valued `arguments`. Responses with
explanatory or trailing text remain invalid. This deterministic normalization
was likewise fixed during compatibility testing, before live-300 generation.

For official scoring, the immutable raw result files are copied to a separate
scorer-input directory. Deterministically extracted singleton calls are mapped
from `{"name": f, "arguments": a}` to the decoded AST form `{f: a}` required
by BFCL's `is_function_calling_format_output` check. The installed Granite
handler does not perform this final mapping, creating an internal interface
mismatch in `bfcl-eval==2026.3.23`. No function name or argument value is
changed. Unparseable raw strings remain unchanged and therefore remain scorer
errors. The scorer is invoked through BFCL's registered Granite 3.x handler
alias; reports and manuscript text retain the actual Granite 3.3 2B model
identity.

Schema-constrained calls receive the same Granite-native prompt plus the
previously defined constrained-output instruction. The underlying llama.cpp
JSON Schema mechanism and CoF activation rule are unchanged.

## Compatibility-12 gate

The explicit development IDs are frozen in `compatibility12.json`: four Java,
four JavaScript, and four multiple-function-selection tasks. They are part of
the earlier development pool and are excluded from addendum inference.

Run D0, AC1, DR1, and COF1 to verify:

- load and request completion;
- Granite prompt and response compatibility;
- at least one parser-readable unconstrained tool call;
- constrained response compatibility;
- checkpoint, token, and latency fields.

This gate has no BFCL-accuracy threshold. Outputs cannot be used to tune
prompt wording.

## Robustness-300 run

After compatibility passes, reuse the exact IDs, hashes, seeds, and generation
ceiling in `experiments/cof_confirmatory_live300/config.json`. Run D0, AC1,
DR1, and COF1 once. Do not exclude, patch, or selectively rerun an item.

The accepted answers have already been accessed for the completed Qwen study,
so this extension is not blinded or prospectively confirmatory. Fixed code,
item IDs, and no-tuning rules limit outcome-aware flexibility.

## Stop rule

After the four Granite conditions and predefined analyses, stop experiments
regardless of result direction. A documented critical implementation defect
may trigger a complete affected-condition restart; poor model accuracy may
not.
