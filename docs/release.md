# Release Checklist

## One-Time Setup

1. Create the GitHub repository:

   ```bash
   git remote add origin git@github.com:szhan311/adm-monai-unet.git
   git push -u origin main
   ```

2. Create the PyPI project with Trusted Publishing:
   - Project name: `adm-monai-unet`
   - Owner: `szhan311`
   - Repository: `adm-monai-unet`
   - Workflow: `publish.yml`
   - Environment: `pypi`

3. In GitHub repository settings, create an environment named `pypi`.

## Local Preflight

```bash
python -m pip install -e ".[dev,experiments]"
ruff check .
pytest -q
python -m build
python -m twine check dist/*
```

## Release

1. Update `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Commit the release:

   ```bash
   git commit -am "Release v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```

4. Create a GitHub release from the tag. The `Publish` workflow will build and
   upload the package to PyPI.

## Docker

Build a local image:

```bash
docker build -t adm-monai-unet:latest .
docker run --rm adm-monai-unet:latest
```
