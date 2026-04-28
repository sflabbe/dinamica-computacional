
.PHONY: sync test test-fast lock lock-check smoke clean

sync:
	uv sync --all-extras --dev

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow"

lock:
	uv lock

lock-check:
	uv lock --check

smoke:
	uv run python tools/smoke_test.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info outputs
