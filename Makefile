SHELL = /bin/bash

.PHONY: get-opik

pull-opik:
	bash scripts/pull_opik.sh

env-setup: get-opik
	pixi install
	pixi run pip install --upgrade -r requirements.txt

docker-up:
	docker compose -f infra/docker-compose.yaml --profile opik up -d --build

docker-down:
	docker compose -f infra/docker-compose.yaml --profile opik down

fastapi-app:
	uvicorn api:app --reload --port 8001

streamlit-app:
	streamlit run app.py --server.port 8501
