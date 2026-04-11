# weiss_schwarz_rl

Thesis RL pipeline scaffold for Weiss Schwarz.

This repo is strongest at config, contracts, provenance, and small standalone components. It also includes a minimal inline training smoke path plus a clearly labeled public-safe toy/demo pipeline for CI and artifact checks. The full thesis runtime still lives behind the dedicated entrypoints and is described honestly in the docs.

## Start here

Read these first:

- [Docs hub](docs/README.md)
- [Getting started](docs/getting_started.md)
- [Runtime modes](docs/runtime_modes.md)
- [Artifact contract](docs/artifact_contract.md)
- [Troubleshooting](docs/troubleshooting.md)

## Supported install paths

```bash
uv sync --extra dev
```

Optional simulator package extra:

```bash
uv sync --extra dev --extra sim
```

If you are not using `uv`, install the editable package with dev extras:

```bash
python -m pip install -e ".[dev]"
```

## Fast verification

```bash
make verify
bash scripts/run_local_ci_parity.sh
```

## Working with runs

The current repo keeps the public-safe demo pipeline and the thesis-oriented pipeline separate on purpose.

```bash
make train-min
make train-inline-smoke
make toy-public-e2e
make artifact-hygiene
```

The demo pipeline is synthetic and public-safe. The thesis-oriented run paths are described in `docs/runtime_modes.md` and `docs/artifact_contract.md`.
