# BFCL Language-Semantics Alignment Note

Date: 2026-07-27  
Scope: v3 `cof_pilot228`

## Issue

The first 0.6B technical run applied native JSON types to all BFCL datasets.
That interpretation is correct for Python-style calls but not for BFCL Java
and JavaScript calls. In those two datasets, BFCL expects argument values to be
JSON strings encoding language literals. For example, an array parameter is
represented as the string `'["x"]'`, not as a native JSON array.

This mismatch created false validator triggers and made some constrained
outputs incompatible with the official BFCL scorer. A concrete audit case was
`simple_javascript_39`: the Direct output used BFCL's expected string
representation, while the old validator incorrectly rejected it and a
constrained retry converted the value to a native array.

## Correction

- The validator now accepts a `language` parameter and normalizes Java and
  JavaScript argument schemas to string-valued properties.
- The constraint compiler applies the identical language-aware normalization.
- The runner derives the language from the frozen dataset name and passes it to
  both components.
- Regression tests cover JavaScript string-encoded collection arguments.
- All 228 generated schemas pass JSON Schema Draft 2020-12 validation.

## Data handling

The original 0.6B artifacts are preserved under
`runs/0.6b_pre_language_fix/` as a technical audit trail. Their policy
summaries are invalid for scientific interpretation and will not be reported
as study results.

The Direct response text is model output unaffected by the faulty validator,
so it is reused byte-for-byte and revalidated under the corrected semantics.
Every recovery or always-constrained condition is regenerated under the
corrected semantics.

## Integrity consequence

No conclusion from the pre-fix recovery run is retained. Only results produced
after this alignment correction are eligible for analysis or inclusion in the
manuscript.
