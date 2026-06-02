
# =========================
# Config
# =========================

PYTHON = python
QUARTO = quarto

FIG_SCRIPTS = \
	scripts/test.py \
	scripts/test2.py
	
# =========================
# Top-level targets
# =========================

.PHONY: all figures site clean help

## Default: build everything
all: figures site

## Build only figures
figures:
	@echo "==> Building figures..."
	@for script in $(FIG_SCRIPTS); do \
		echo "Running $$script"; \
		$(PYTHON) $$script || exit 1; \
	done

## Render Quarto site + PDFs
site:
	@echo "==> Rendering Quarto project..."
	$(QUARTO) render

## Clean outputs
clean:
	@echo "==> Cleaning generated files..."
	find lectures -name "*.pdf" -delete
	find lectures -name "figures" -type d -exec rm -rf {} +
	rm -rf _site

## Help
help:
	@echo "Available targets:"
	@echo "  make all      - build figures + render site"
	@echo "  make figures  - only generate figures"
	@echo "  make site     - only run quarto render"
	@echo "  make clean    - remove generated outputs"