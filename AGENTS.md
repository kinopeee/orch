# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**orch** is a Python CLI task orchestrator that executes CLI agents as a DAG. It is a single Python package (not a monorepo) with no external service dependencies — all state is file-based on the local filesystem (`.orch/` directory).

### Prerequisites

- Python ≥3.11 (the VM has Python 3.12.3)
- `python3.12-venv` system package is required for `python3 -m venv` (not installed by default on Ubuntu — the update script handles the venv setup)

### Development commands

See `README.md` for full reference. Quick summary:

```bash
source .venv/bin/activate
ruff format --check .   # format check
ruff check .            # lint
mypy src                # type check (strict mode)
pytest                  # 1501 tests, ~3 min
```

### Running the application

```bash
orch run examples/plan_basic.yaml
orch run examples/plan_parallel.yaml --max-parallel 2
orch status <run_id>
orch logs <run_id> --task <task_id> --tail 50
```

### Non-obvious notes

- The `pytest` suite takes approximately **3 minutes** to run (1501 tests). Many tests exercise real subprocess execution and timeouts, so this is expected.
- `python tools/dod_check.py` runs a comprehensive end-to-end smoke test across all example plans. Add `--skip-quality-gates` to skip lint/type checks and only validate runtime behavior.
- The `.orch/` directory is created at the working directory root during execution. It is gitignored and safe to delete to reset state.
- Always activate the venv (`source .venv/bin/activate`) before running any dev commands.
