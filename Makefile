.PHONY: setup download clean help

help:
	@echo "UMinFramework - Available Make Targets:"
	@echo ""
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  download   - Download and preprocess datasets (runs setup if needed)"
	@echo "  download-dev - Download using system Python (for development/testing)"
	@echo "  clean      - Remove virtual environment and generated data files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Quick start: make download"

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete!"

download: .venv/bin/python
	@echo "Running dataset download and preprocessing..."
	./.venv/bin/python scripts/download_datasets.py
	@echo "✓ Download complete! Check data/ directory for results."

download-dev:
	@echo "Running dataset download with system Python (development mode)..."
	python3 scripts/download_datasets.py
	@echo "✓ Download complete! Check data/ directory for results."

.venv/bin/python:
	@echo "Virtual environment not found, running setup..."
	$(MAKE) setup

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -f data/*.jsonl data/*_sample.json data/*_info.txt
	@echo "✓ Cleanup complete!"
