# Paper 1: Predicting Cross-Asset Stablecoin Contagion with Temporal GNNs

**Compiled PDF: `main.pdf`** (built locally with TeX Live 2026 / `acmart`, 0 errors).

Self-contained: `main.tex` + `figures/` + `references.bib`. ACM `sigconf` (`nonacm` draft mode).

## Build
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
or upload this folder to Overleaf.

Every number is reproduced by `../../reproduce.sh`. The exported hub ranking is the input to the
companion causal-validation paper (`stablecoin-abm/paper/standalone_abm_paper/`).
