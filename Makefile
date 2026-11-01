.DEFAULT_GOAL=help

# Required for globs to work correctly
SHELL := /bin/bash
VERSION := $(shell git describe --tags --always)

.EXPORT_ALL_VARIABLES:

.PHONY: all
all: build  ## build all default targets

.PHONY: build
build:  ## build image
	docker build . \
		-f Dockerfile \
		--platform linux/amd64 \
		--build-arg VERSION=$(VERSION) \
		-t hangxie/ray-vllm:$(VERSION)

.PHONY: help
help:  ## Print list of Makefile targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		cut -d ":" -f1- | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
