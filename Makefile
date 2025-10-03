SHELL = /bin/bash

env-setup-dev:
	pixi install

.PHONY: pull-opik
pull-opik:
	bash scripts/pull_opik.sh

.PHONY: docker-up
docker-up: pull-opik
	docker compose -f infra/docker-compose.yaml --profile opik --profile app up -d --build

.PHONY: docker-down
docker-down:
	docker compose -f infra/docker-compose.yaml --profile opik --profile app down
