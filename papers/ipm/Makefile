LATEXMK ?= latexmk
PYTHON ?= python

.PHONY: all assets pdf clean

all: assets pdf

assets:
	$(PYTHON) scripts/generate_assets.py

pdf:
	LC_ALL=C LANG=C $(LATEXMK) -pdf -interaction=nonstopmode -file-line-error main.tex

clean:
	$(LATEXMK) -C main.tex
