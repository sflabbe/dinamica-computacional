.PHONY: sync test test-no-numba test-fast lock lock-check smoke smoke-app app clean

sync:
	uv sync --all-extras --dev

test-no-numba:
	DC_USE_NUMBA=0 PYTHONPATH=src:. pytest -q

test:
	PYTHONPATH=src:. pytest -q

test-fast:
	uv run pytest -q -m "not slow"

app:
	uv run --extra app streamlit run app/main.py

smoke-app:
	PYTHONPATH=src:. python tools/smoke_app_services.py

lock:
	uv lock

lock-check:
	uv lock --check

smoke:
	uv run python tools/smoke_test.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info outputs
