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
    uvx twine check --strict dist/*
    uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py

# Check the library works on an older Python (default 3.12): runs the whole test suite against it
check-python version="3.12":
    #!/usr/bin/env bash
    set -euo pipefail
    workdir=$(mktemp -d)
    trap 'rm -rf "$workdir"' EXIT
    tar -cf - --exclude=__pycache__ pyproject.toml README.md LICENSE src tests uv.lock $([ -f .env ] && echo .env) | tar -xf - -C "$workdir"
    sed -i 's/^requires-python = .*/requires-python = ">={{ version }}"/' "$workdir/pyproject.toml"
    cd "$workdir"
    uv run --python {{ version }} pytest

# Run all checks
checks: lint format type test smoke

# Release a new version: `just release minor` or `just release 0.3.0`
release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Preflight
    [[ -z "$(git status --porcelain)" ]] || { echo "Working tree is dirty"; exit 1; }
    [[ "$(git branch --show-current)" == "main" ]] || { echo "Not on main"; exit 1; }
    git pull --ff-only

    # Ensure everything ok
    just checks
    # Ensure clean tree after checks ran
    git diff --exit-code

    # Cleanup on errors and on Ctrl-C. Only delete the tag this run created: re-releasing
    # an already released version fails before `git tag`, and must not delete that tag.
    start=$(git rev-parse HEAD)
    created_tag=0
    cleanup() {
        if [[ "${created_tag}" == 1 ]]; then
            git tag -d "v${new}" >/dev/null
        fi
        git reset --hard "$start" >/dev/null 2>&1 || true
    }
    trap cleanup ERR
    trap 'cleanup; exit 130' INT

    # bump
    if [[ "{{ version }}" =~ ^[0-9] ]]; then
        uv version "{{ version }}"
    else
        uv version --bump "{{ version }}"
    fi
    new=$(uv version --short)

    # Nothing to release: `uv version` is a no-op when the version is unchanged, so this
    # would otherwise fail later on an empty commit
    if git rev-parse -q --verify "refs/tags/v${new}" >/dev/null; then
        echo "v${new} is already tagged"
        cleanup  # `exit` does not fire the ERR trap, so revert the bump by hand
        exit 1
    fi

    # Tag and push
    git commit -am "chore: release v${new}"
    git tag "v${new}"
    created_tag=1
    git push --atomic origin main "v${new}"
    trap - ERR INT

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
