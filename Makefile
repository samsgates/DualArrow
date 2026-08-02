.PHONY: install test lint format typecheck run ingest query docker-up docker-down

install:
	python -m pip install -e ".[dev,all-adapters]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy dualarrow

run:
	dualarrow serve --config dualarrow.yaml

ingest:
	dualarrow ingest --config dualarrow.yaml

query:
	dualarrow query "Which bearing is approved for P-104?" --config dualarrow.yaml

docker-up:
	docker compose --profile production up --build

docker-down:
	docker compose --profile production down
