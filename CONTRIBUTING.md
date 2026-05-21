# Contributing

Contributions are welcome. Keep changes focused and include tests for behavior
that affects the public API.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,experiments]"
```

## Checks

Run these before opening a pull request:

```bash
ruff check .
pytest -q
adm-unet-experiments --output-dir results/local/smoke --steps 1 --eval-batches 1 --batch-size 1
```

## Pull Requests

- Explain the user-visible behavior change.
- Include benchmark or smoke-test output when changing model code.
- Do not commit generated caches, local environments, or large checkpoints.
- Keep vendored ADM code attribution intact.
