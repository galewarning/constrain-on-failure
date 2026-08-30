# CoF Confirmatory Live-300 Protocol

Frozen: 2026-07-27  
Status: confirmatory; no generation inspected at freeze time

## Objective

Test whether a schema-constrained retry after a validator-detectable Direct
failure improves official BFCL correctness over a budget-matched diagnostic
retry on previously unused live tasks.

## Frozen sample

- 150 of 258 entries from `BFCL_v4_live_simple.json`
- 150 of 1,053 entries from `BFCL_v4_live_multiple.json`
- 300 unique items per model
- deterministic sampling without replacement with seed `20260728`
- explicit selected IDs stored in `config.json`
- no overlap with the non-live development pilot

Source SHA-256:

- live simple:
  `1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4`
- live multiple:
  `fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8`
- frozen `config.json`:
  `c2a4d17d6c99fc825b86588f936da88a01ae3d35bc9e149a77f1bcdbdad2e06f`

The runner verifies source hashes before loading any item. The manifest-freeze
script reads only prompt/function data files and does not locate or open
possible-answer files.

## Models and runtime

- Qwen3-0.6B Q8_0 GGUF, revision
  `23749fefcc72300e3a2ad315e1317431b06b590a`
- Qwen3-1.7B Q8_0 GGUF, revision
  `90862c4b9d2787eaed51d12237eafdfe7c5f6077`
- llama.cpp b10141, CPU, 12 threads, 4,096-token context
- temperature zero, item seeds 4000–4299, 512-token output ceiling

## Conditions

| ID | Policy | Maximum generations |
|---|---|---:|
| D0 | unconstrained Direct | 1 |
| AC1 | always-constrained initial generation | 1 |
| DR1 | diagnostic retry after invalid Direct | 2 |
| COF1 | constrained retry after invalid Direct | 2 |

D0 is generated once per model and reused byte-for-byte by DR1 and COF1.
DR1 and COF1 therefore have identical maximum generation budgets.

## Outcomes

Primary:

- official BFCL correctness for COF1 versus DR1.

Secondary:

- COF1 versus AC1 and D0;
- schema validity and wrong-valid rate;
- helpful and harmful paired changes;
- token and latency overhead;
- model- and category-specific effects.

## Confirmatory analysis

- paired accuracy difference;
- 10,000-resample bootstrap clustered by BFCL item across models;
- per-model exact McNemar tests as secondary analyses;
- report effect size and confidence interval regardless of direction.

The primary claim is confirmed when the combined COF1-minus-DR1 estimate is
positive, neither model has a negative estimate, and no integrity defect
invalidates the comparison. A clustered 95% interval above zero is strong
confirmation; a positive interval crossing zero is directional but
inconclusive.

## Integrity embargo and no-tuning rule

- BFCL possible answers may be accessed only after both models and all four
  conditions finish generation.
- Before that point, only prompt files, raw generations, validator outcomes,
  token counts, latency, and filesystem/format checks may be inspected.
- Prompts, validator behavior, constraints, seeds, parameters, and policies are
  frozen.
- Generated failures are retained and not patched.
- A critical implementation defect requires a documented restart of every
  affected condition.
