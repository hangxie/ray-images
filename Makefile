.DEFAULT_GOAL=help

# Required for globs to work correctly
SHELL := /bin/bash
VERSION := $(shell git describe --tags --always)
# --label as well as the Dockerfile LABEL: Podman's layer cache reuses the final LABEL layer when only VERSION changes

.EXPORT_ALL_VARIABLES:

.PHONY: all
all: build  ## build all default targets

.PHONY: build
build:  ## build hangxie/ray-images
	docker build . \
		-f Dockerfile \
		--platform linux/amd64 \
		--build-arg VERSION=$(VERSION) \
		--label org.opencontainers.image.version=$(VERSION) \
		-t hangxie/ray-images:$(VERSION)

.PHONY: help
help:  ## Print list of Makefile targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		cut -d ":" -f1- | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
