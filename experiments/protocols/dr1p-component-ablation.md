# DR1-P Subsequent Component Ablation

Status: completed subsequent/post-hoc analysis; not part of prospective RQ1.

## Purpose

The original DR1-to-COF1 comparison changes both retry wording and decoder
enforcement. DR1-P reuses the frozen live-300 D0 outputs and separates those
components without changing any historical condition:

- DR1-P minus DR1 estimates the added retry-instruction contribution.
- COF1 minus DR1-P estimates the incremental hard-decoder contribution after
  wording, context, endpoint, wrapper, seed, and generation parameters are
  matched.
- COF1 minus DR1 remains the original prompt-plus-decoder bundle comparison.

## Policy

For each model, DR1-P starts from the exact stored D0 generation. A D0-valid
row is returned unchanged and generates no request. A D0-invalid row receives
the same rejected call and deterministic validator diagnosis as DR1 and COF1,
followed by the exact frozen COF1 instruction:

> For this constrained response, return only one JSON object with exactly the
> keys name and arguments. Preserve the user's requested values. Do not use XML
> tags or explanatory text.

The retry uses the same llama.cpp `/completion` endpoint and runner-applied
tool-call wrapper as COF1, but the request payload omits `json_schema`.

## Frozen reuse gate

`results/aggregate/dr1p-manifest.json` verifies the 300 ordered task IDs, source-config hash, D0/DR1/
COF1 checkpoint hashes, byte-identical stored D0 initial outputs, and identical
DR1/COF1 trigger sets. Verified retry counts were 66 for Qwen3-0.6B, 38 for
Qwen3-1.7B, and 33 for Granite 3.3 2B, for 137 new requests.

No D0, AC1, DR1, or COF1 generation was regenerated or overwritten. No
possible-answer content entered a generation prompt, schema, validator message,
or retry context.

## Implementation correction before use

An initial candidate implementation used a different completion interface and
did not match COF1's runner-applied wrapper. The candidate run was quarantined
in the internal study record and excluded from this release artifact. No
candidate statistic entered the manuscript. The corrected arm was tested and
all 137 retries were rerun as one complete affected condition. This correction
does not affect the frozen historical policies.

## AC1 audit

`results/aggregate/ac1-invalid-audit-summary.json` records that all three
AC1-invalid model rows trace to the same
BFCL task. JSON Schema `number` admits integer-valued JSON numbers, whereas the
BFCL-aligned validator requires exact float instances in nested float arrays.
The complete outputs were well below the token ceiling. The finding does not
invalidate historical scores.

## Release-artifact outputs

- `results/aggregate/dr1p-analysis.json`: paired effects, bootstrap intervals,
  exact McNemar results, combined descriptive averages, and the triggered-subset
  mechanism table.
- `results/aggregate/dr1p-manifest.json`: frozen-input and trigger provenance.
- `results/aggregate/dr1p-verification.json`: fail-closed verification summary.
- `results/aggregate/ac1-invalid-audit-summary.json`: aggregate deterministic
  AC1 failure-audit conclusion.

Raw generations, item-level outputs, runtime logs, third-party benchmark data,
and the quarantined candidate run are intentionally excluded.
