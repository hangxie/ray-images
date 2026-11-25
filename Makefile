.DEFAULT_GOAL=help

# Required for globs to work correctly
SHELL := /bin/bash
MARKDOWN := $(shell git ls-files '*.md')
VERSION := $(shell git describe --tags --always)
# --label as well as the Dockerfile LABEL: Podman's layer cache reuses the final LABEL layer when only VERSION changes

.EXPORT_ALL_VARIABLES:

# all's steps share files and the venv: format rewrites what lint, test, and build read
.NOTPARALLEL:

.PHONY: all
all: dev format lint test pins build  ## Build all common targets and the image

.PHONY: dev
dev:  ## Install the development environment (.venv) from uv.lock
	@echo "==> Installing the development environment"
	@uv sync --dev --locked

.PHONY: lock
lock:  ## Re-resolve uv.lock after changing pyproject.toml
	@echo "==> Re-resolving uv.lock"
	@uv lock

.PHONY: format
format:  ## Format all python code and apply safe lint fixes
	@echo "==> Formatting all python code"
	@uv run --locked ruff format .
	@uv run --locked ruff check --fix .

.PHONY: lint
lint:  ## Run static code analysis
	@echo "==> Running static code analysis"
	@echo "==> Running ruff format --check ..."
	@uv run --locked ruff format --check .
	@echo "==> Running ruff check ..."
	@uv run --locked ruff check .
	@echo "==> Running mypy ..."
	@uv run --locked mypy
	@echo "==> Running Markdown check ..."
	@uv run --locked python scripts/check_markdown.py $(MARKDOWN)

.PHONY: test
test:  ## Run unit tests (CPU)
	@echo "==> Running unit tests"
	@uv run --locked pytest

.PHONY: pins
pins:  ## Check the dev environment against vLLM's pins on PyPI
	@echo "==> Checking pins against vLLM on PyPI"
	@uv run --locked pytest -m network

.PHONY: check
check: lint test  ## Lint and test, offline (CI runs this and pins)

.PHONY: build
build:  ## build hangxie/ray-images
	docker build . \
		-f Dockerfile \
		--platform linux/amd64 \
		--build-arg VERSION=$(VERSION) \
		--label org.opencontainers.image.version=$(VERSION) \
		-t hangxie/ray-images:$(VERSION)

.PHONY: clean
clean:  ## clean up temporary files
	@echo "==> Removing temporary files from build and test"
	@rm -rf -- .venv .coverage
	@find . \( -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache -o -name __pycache__ \) \
		-type d -prune -exec rm -rf -- {} +

.PHONY: help
help:  ## Print list of Makefile targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		cut -d ":" -f1- | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
