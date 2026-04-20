.PHONY: sync test

sync:
	uv sync && \
	uv pip install --no-deps --no-build-isolation \
		"promptbench @ git+https://github.com/microsoft/promptbench.git@fcda538bd779ad11612818e0645a387a462b5c3b"

test:
	uv run pytest tests/ -v
