SHELL := /usr/bin/env bash
TEX := report.tex

.PHONY: pdf clean watch

pdf:
	latexmk -xelatex -shell-escape -file-line-error -interaction=nonstopmode $(TEX)

clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.fls *.fdb_latexmk

watch:
	latexmk -xelatex -shell-escape -pvc -interaction=nonstopmode $(TEX)


