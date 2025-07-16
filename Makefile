install-dev:
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov isort bandit safety

test:
	pytest

test-cov:
	pytest --cov=radar_evaluator --cov-report=term-missing --cov-report=xml --cov-report=html

lint:
	black --check .
	flake8 .
	mypy .

format:
	black .
	isort .

check-security:
	bandit -r .
	safety check

clean:
	rm -rf .pytest_cache .mypy_cache htmlcov coverage.xml .venv 