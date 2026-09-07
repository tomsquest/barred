# List available commands
default:
    @just --list

# Run linter and apply automatic fixes
lint:
    uv run ruff check --fix --show-fixes

# Format code
format:
    uv run ruff format

# Check types
type:
    uv run ty check

# Run tests
test *args:
    uv run pytest {{ args }}

# Run tests involving an LLM (excluded from normal test run)
test-llm *args:
    uv run pytest -m llm -s {{ args }}

# Build the wheel and check it is importable by a consumer, in an isolated env
smoke:
    rm -rf dist
    uv build
    uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py

# Check the library works on an older Python (default 3.12): runs the whole test suite against it
check-python version="3.12":
    #!/usr/bin/env bash
    set -euo pipefail
    workdir=$(mktemp -d)
    trap 'rm -rf "$workdir"' EXIT
    tar -cf - --exclude=__pycache__ pyproject.toml README.md LICENSE src tests demo uv.lock $([ -f .env ] && echo .env) | tar -xf - -C "$workdir"
    sed -i 's/^requires-python = .*/requires-python = ">={{ version }}"/' "$workdir/pyproject.toml"
    cd "$workdir"
    uv run --python {{ version }} pytest

# Run all checks
checks: lint format type test smoke

# Install dependencies and set up the project (run this after cloning the repo)
install:
    uv sync
    uv run prek install

# Upgrade dependencies, precommits and GitHub Actions
upgrade:
    uv lock --upgrade
    uv run prek update --freeze --cooldown-days 7
    npx -y actions-up@latest

# Remove all generated files and reset the project to a clean state
clean:
    uvx cleanpy -a .
    uv sync

# Generate the dimensions and instantiations from the Criterion and Query-Designation pairs
generate-dimensions *args:
    uv run python -m demo.main generate-dimensions {{ args }}

# Generate the samples from the dimensions/instantiations
generate-samples *args:
    uv run python -m demo.main generate-samples {{ args }}

# Generate the dataset in CSV format
generate-dataset *args:
    uv run python -m demo.main generate-dataset {{ args }}
