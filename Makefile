.PHONY: help env install install-pip test lint build clean

help:
	@echo "kinapse make targets:"
	@echo "  make env          create/update the conda env (also installs kinapse)"
	@echo "  make install      editable install with all pip extras (into current env)"
	@echo "  make install-pip  editable install, core + numbering + reduce (no torch)"
	@echo "  make test         run the smoke test suite"
	@echo "  make lint         ruff check the package"
	@echo "  make build        build sdist + wheel"
	@echo "  make clean        remove build/cache artifacts"

env:
	./setup.sh

install:
	python -m pip install -e ".[all,dev]"

install-pip:
	python -m pip install -e ".[numbering,reduce,dev]"

test:
	pytest -q

lint:
	ruff check src tests

build:
	python -m build

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
