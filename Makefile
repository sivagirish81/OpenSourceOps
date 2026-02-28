PYTHON := $(shell command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup init_snowflake run test

setup:
	@$(PYTHON) -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,13), 'Python 3.10-3.12 required for current dependency set. Install/use python3.11 or python3.12, then rerun make setup.'"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

init_snowflake:
	$(PY) -c "from storage.snowflake_store import SnowflakeStore; s=SnowflakeStore(); s.ensure_schema(); print('Snowflake schema initialized')"

run:
	$(VENV)/bin/streamlit run app/main.py

test:
	$(VENV)/bin/pytest -q
