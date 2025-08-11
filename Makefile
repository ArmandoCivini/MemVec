.PHONY: help install test clean docker-up docker-down lint format

help: ## Show this help message
	@echo "MemVec - Development Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and setup development environment
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install pytest black flake8 mypy

test: ## Run all tests
	.venv/bin/pytest tests/

test-unit: ## Run unit tests only
	.venv/bin/pytest tests/ -m "not integration"

test-integration: ## Run integration tests only
	.venv/bin/pytest tests/ -m "integration"

lint: ## Run code linting
	.venv/bin/flake8 src/ tests/
	.venv/bin/mypy src/

format: ## Format code with black
	.venv/bin/black src/ tests/

docker-up: ## Start Redis with docker-compose
	docker-compose up -d

docker-down: ## Stop Redis containers
	docker-compose down

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

dev-setup: install docker-up ## Complete development setup
	@echo "Development environment ready!"
	@echo "Activate venv with: source .venv/bin/activate"
