.PHONY: install download-data eda train evaluate test run-api run-frontend docker-build docker-up

VENV = .venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
PYTEST = $(VENV)/Scripts/pytest

install:
	py -3.13 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download-data:
	$(PYTHON) scripts/download_data.py

eda:
	$(PYTHON) scripts/run_eda.py

train:
	$(PYTHON) cli.py train

evaluate:
	$(PYTHON) cli.py evaluate

test:
	$(PYTEST) -v tests/

run-api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

docker-build:
	docker compose build

docker-up:
	docker compose up
