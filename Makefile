HELL = /bin/bash

env-setup:
	pixi install
	pixi run pip install --upgrade -r requirements.txt

docker-up:
	docker compose -f infra/docker-compose.yaml up -d --build

docker-down:
	docker compose -f infra/docker-compose.yaml down
