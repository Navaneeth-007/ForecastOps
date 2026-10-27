.PHONY: install train api dashboard test lint demo
install:
	python -m pip install -e '.[dev]'
train:
	python -m scripts.train
api:
	uvicorn app.main:app --host 127.0.0.1 --port 8000
dashboard:
	streamlit run frontend/streamlit_app.py
test:
	pytest --cov=app --cov-report=term-missing
lint:
	ruff check .
	ruff format --check .
demo:
	docker compose up --build
