# Constrain-on-Failure

Reference implementation and aggregate results for **Constrain-on-Failure (CoF)**, a selective retry policy for structured tool calling with small, locally hosted language models.

CoF begins with an unconstrained generation. If the output is not a valid function call, it retries once with a JSON Schema grammar derived from the available tool definitions. The repository also implements the comparison policies used in the study:

- **D0:** direct generation without retry.
- **AC1:** schema-constrained generation from the first attempt.
- **DR1:** one unconstrained retry with a diagnostic message.
- **DR1-P:** a post-hoc instruction-matched unconstrained retry used to
  separate retry-wording effects from hard-decoder effects.
- **CoF1:** one schema-constrained retry only after the first attempt fails validation.

## Repository contents

- `src/schemahint_agent/`: schema compilation and function-call validation.
- `experiments/`: frozen runner, BFCL integration, and statistical analysis scripts.
- `experiments/protocols/`: confirmatory, robustness, and subsequent component-ablation protocols recorded during the study.
- `results/aggregate/`: aggregate statistical outputs and validation summaries
  reported by the study.
- `tests/`: unit tests for the core logic and runner transformations.

This release artifact intentionally excludes model weights, BFCL source/answer data, item-level prompts and generations, runtime logs, quarantined development runs, and manuscript working files. Obtain BFCL through its official distribution and obtain model weights from their respective publishers.

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Install the optional benchmark dependencies only when reproducing BFCL runs:

```bash
python -m pip install -e ".[benchmark]"
```

## Verify the implementation

```bash
python -m pytest
```

## Reproducing the experiments

1. Install the benchmark dependencies and a compatible local `llama.cpp` server. To run the complete test suite as well, install `.[dev,benchmark]`.
2. Obtain BFCL through its official package. Do not copy benchmark answer files into this repository.
3. Read the matching document in `experiments/protocols/` and use its frozen file in `experiments/configs/` before running a condition.
4. Run `python experiments/run_cof_pilot.py --help` for the full runner interface.
5. Use the analysis scripts in `experiments/` to regenerate aggregate statistics from locally generated BFCL score artifacts. The DR1-P arm reuses the frozen D0 outputs and generates only for the 137 D0-invalid model-task observations documented in its manifest.

The frozen confirmatory configuration includes benchmark file hashes and selected identifiers but does not redistribute the benchmark items themselves.

## Data and code availability

The source code, protocols, tests, and non-restricted aggregate results are provided under the Apache License 2.0. Third-party datasets, benchmark answers, model artifacts, and their licenses are not covered by this repository.

## Citation

The manuscript citation and archival DOI will be added after publication metadata and the first archived release are available.
