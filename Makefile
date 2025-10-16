SHELL := /usr/bin/env bash
TEX := report.tex

# Detect latexmk availability
HAS_LATEXMK := $(shell command -v latexmk 2>/dev/null)
# Detect tlmgr availability (TeX Live manager)
HAS_TLMGR := $(shell command -v tlmgr 2>/dev/null)

.PHONY: pdf clean watch check deps

pdf:
ifeq ($(HAS_TLMGR),)
	@echo "tlmgr not found; skipping TeX package install"
else
	$(MAKE) deps
endif
ifeq ($(HAS_LATEXMK),)
	# Fallback build without latexmk
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	biber $(basename $(TEX)) || true
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
else
	latexmk -xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
endif

clean:
ifeq ($(HAS_LATEXMK),)
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.toc *.xdv
else
	latexmk -C
	rm -f *.bbl *.run.xml *.fls *.fdb_latexmk
endif

watch:
ifeq ($(HAS_LATEXMK),)
	@echo "watch requires latexmk; please install latexmk to use this target."
else
	latexmk -xelatex -shell-escape -pvc -interaction=nonstopmode $(TEX)
endif

check:
ifeq ($(HAS_TLMGR),)
	@echo "tlmgr not found; skipping TeX package install"
else
	$(MAKE) deps
endif
ifeq ($(HAS_LATEXMK),)
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.toc *.xdv
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	biber $(basename $(TEX)) || true
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
else
	latexmk -C
	latexmk -xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	biber $(basename $(TEX)) || true
	latexmk -xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
	latexmk -xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)
endif

deps:
ifeq ($(HAS_TLMGR),)
	@echo "tlmgr not found; cannot auto-install TeX packages"
else
	@echo "Installing required TeX packages via tlmgr..."
	tlmgr update --self || true
	tlmgr install biblatex biber biblatex-ieee csquotes siunitx circuitikz lm pgfplots babel-english
endif


