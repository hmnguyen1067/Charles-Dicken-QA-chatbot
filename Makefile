HELL = /bin/bash

env-setup:
	pixi install
	pixi run pip install --upgrade -r requirements.txt
	git clone https://github.com/comet-ml/opik.git

docker-up:
	docker compose -f infra/docker-compose.yaml --profile opik up -d --build

docker-down:
	docker compose -f infra/docker-compose.yaml --profile opik down
