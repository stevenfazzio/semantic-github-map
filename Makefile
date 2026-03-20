.PHONY: install lint format test pipeline poc-reumap clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

pipeline:
	uv run python pipeline/00_enumerate_repos.py
	uv run python pipeline/01_fetch_repos.py
	uv run python pipeline/02_select_top_repos.py
	uv run python pipeline/03_summarize_readmes.py
	uv run python pipeline/04_embed_readmes.py
	uv run python pipeline/05_reduce_umap.py
	uv run python pipeline/06_label_topics.py
	uv run python pipeline/07_visualize.py

poc-reumap:
	uv run python experiments/poc_reumap_precompute.py
	@echo "Open http://localhost:8000/experiments/poc_reumap.html"
	uv run python -m http.server 8000

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
