# Contributing

## Setup

```bash
uv sync
cp .env.example .env   # fill in API keys if you'll be running experiments
```

`uv sync` installs `src/sycophancy` itself as an editable package, so
`from sycophancy.<module> import ...` resolves from anywhere in the repo.

## Where things live

See the "Repository layout" section in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
In short: library code goes in `src/sycophancy/`, anything that calls a
model API goes in `scripts/`, anything that only reads `experiment_out/`
and produces a table/figure/stat goes in `analysis/`.

## Tests

```bash
uv run pytest
```

Tests must not require API keys or make network calls — `scripts/` CLIs are
tested for import-cleanliness only; the actual logic worth unit-testing
(entropy math, the cache-control wrapper, the empty-response backfill
semantics) lives in pure functions in `src/sycophancy/`. If you add a new
pure function with a non-obvious contract, add a test for it.

## Lint

```bash
uv run ruff check src scripts analysis tests
```

Configured to correctness rules only (`E`, `F`) — this is a research
codebase with a lot of pre-existing style that predates this config, so
lint is advisory, not a gate.

## Changing experiment code

If you change anything under `src/sycophancy/` that affects a class whose
instances get pickled (most notably `reasoning_uncertainty.py`'s
dataclasses), check `src/sycophancy/__init__.py`'s pickle-compat shim
still covers it — see the note in `REPRODUCIBILITY.md`. Existing
`experiment_out/**/*.pkl` files are expensive (real API spend) to
regenerate; don't make a change that silently strands them.

If your change affects a number, table, or figure that appears in the
paper, update the corresponding row in `REPRODUCIBILITY.md`'s artifact →
script mapping and re-verify the number, don't just assume the surrounding
prose is still correct.
