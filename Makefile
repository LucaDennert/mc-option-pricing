.PHONY: install test benchmarks synthetic calibrate exotics all

install:
	pip install -e ".[dev]"

test:
	pytest -q

synthetic:
	python scripts/make_synthetic_surface.py

calibrate: synthetic
	python scripts/calibrate_heston.py

exotics: calibrate
	python scripts/price_exotics.py

benchmarks:
	python scripts/run_benchmarks.py

all: test benchmarks exotics
