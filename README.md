# ETF portfolio assignment

Python >=3.10. Create a local virtual environment, then `python -m pip install -r requirements.txt`.
Run the notebook top to bottom with the kernel cwd set to this bundle. The identical Desktop notebook also supports cwd=Desktop with its relative `etf-portfolio-assignment` subfolder. Outputs are in the selected bundle's `results/`.

Default is explicitly frozen Yahoo adjusted prices in `data/close_adj.csv`. Original download provenance is preserved in `data/download_meta.json`. Setting `REFRESH=True` replaces the data using validated batched Yahoo retries; failures raise, without falling back. Study dates are fixed; changing the study needs explicit date edits.

Verification: `python tests/test_assignment.py`, then `python tests/execute_and_verify.py`. The second command executes in fresh kernels, relocates the bundle with no external repo, disables Yahoo download in frozen mode, independently recomputes key numbers and confirms CSV equivalence. It synchronizes the executed Desktop notebook when run in the supplied Desktop location. Regression RED logs are preserved in tests/. No pytest required.

Report inputs: all results/*.csv plus results/summary.json. Rates, weights and drawdowns are decimals. Full-sample optimization uses annualized log-model moments with nominal 3% hurdle. IS/OOS performance uses the same exact simple-return convention and effective daily 3% cash rate. All values are computed, not hardcoded. There is no ex-post refit or misleading anti-tangency strategy.

## Three-page report

`report/report.pdf` is the compiled analytical report. `report/report.tex` is its source; `report/make_report.py` deterministically generates number macros, tables and five TikZ/PGFPlots charts from the notebook outputs. U = unrestricted, L = long-only. Figure coordinates are computed data, not traced screenshots.

From this directory: `python report/make_report.py`, then `cd report` and `latexmk -pdf report.tex`. Install a TeX distribution with TikZ/PGFPlots, Latin Modern, microtype, caption, booktabs and hyperref. Or use `make report`; `make notebook` executes and verifies the notebook first. A report rebuild needs the bundled results, not a network request.

Files: data/ (frozen input and metadata); notebook; requirements; tests/ (regressions, execution and evidence); results/ (full-precision numeric report inputs and figures); report/ (PDF, source, generator and generated fragments). Original notebook backup remains beside the Desktop notebook, outside this bundle.

The frozen snapshot and relocation checks were executed; the optional live Yahoo refresh was not exercised. Python 3.14 was used for validation; the stated minimum Python version has not been tested on another interpreter.
