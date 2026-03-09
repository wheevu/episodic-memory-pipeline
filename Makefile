# Episodic Memory Pipeline Makefile
# Use this for common development tasks

.PHONY: help install install-dev demo demo-clean test test-slow lint format clean doctor

# Default target
help:
	@echo "Episodic Memory Pipeline - Development Tasks"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install the package in editable mode"
	@echo "  make install-dev   Install with development dependencies"
	@echo ""
	@echo "Demo:"
	@echo "  make demo          Bootstrap demo data and run a simple query"
	@echo "  make demo-clean    Clean existing data, then bootstrap fresh"
	@echo "  make demo-mock     Bootstrap using mock providers (fast, no models)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run fast tests (no model downloads)"
	@echo "  make test-slow     Run all tests including slow embedding tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Auto-format code (ruff)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         Remove generated files and caches"
	@echo "  make doctor        Run system diagnostics"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Demo commands
demo:
	@echo "=== Bootstrapping demo data ==="
	python -m scripts.bootstrap_demo
	@echo ""
	@echo "=== Running sample query ==="
	python cli.py query "What am I learning?"

demo-clean:
	@echo "=== Cleaning existing data ==="
	python -m scripts.bootstrap_demo --clean
	@echo ""
	@echo "=== Running sample query ==="
	python cli.py query "What am I learning?"

demo-mock:
	@echo "=== Bootstrapping with mock providers ==="
	python -m scripts.bootstrap_demo --mock --clean
	@echo ""
	@echo "=== Running sample query (mock) ==="
	python cli.py --mock query "What am I learning?"

# Testing
test:
	pytest tests/ -v

test-slow:
	pytest tests/ -v --run-slow

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not installed. Run: make install-dev"; exit 1; }
	ruff check src/ tests/ cli.py config.py

format:
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not installed. Run: make install-dev"; exit 1; }
	ruff format src/ tests/ cli.py config.py
	ruff check --fix src/ tests/ cli.py config.py

# Diagnostics
doctor:
	python cli.py doctor

doctor-dry:
	python cli.py doctor --dry

# Evaluation
eval:
	python cli.py eval-run --scenario diary

eval-list:
	python cli.py eval-list

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf src/*/*/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	@echo "Cleaned build artifacts and caches"

clean-data:
	rm -f data/*.db data/*.db-shm data/*.db-wal
	rm -f data/*.npy data/*.index
	rm -rf data/lancedb
	@echo "Cleaned generated data files"

clean-all: clean clean-data
	@echo "Cleaned everything"
