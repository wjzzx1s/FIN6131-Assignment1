PYTHON ?= python
.PHONY: notebook report all
notebook:
	$(PYTHON) tests/execute_and_verify.py
report:
	$(PYTHON) report/make_report.py
	cd report && latexmk -pdf -silent -interaction=nonstopmode -halt-on-error report.tex
all: notebook report
