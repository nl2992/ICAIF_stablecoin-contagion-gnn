# Joint paper — LaTeX (ACM `sigconf`)

Submission-ready source for the joint contribution spanning both repos
(`stablecoin-contagion-gnn` + `stablecoin-abm`).

- `main.tex` — ACM `sigconf` (set to `nonacm` for a draft; remove for camera-ready).
- `figures/` — the seven figures referenced, copied from the two repos' results.
- A prose version with the full narrative is `../JOINT_PAPER.md`.

## Build

No LaTeX is installed locally. Easiest path: upload this `paper/` folder to **Overleaf**
(New Project → Upload Project → zip of this folder) and compile `main.tex` with pdfLaTeX.

Locally (if you install MacTeX):
```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Notes for camera-ready
- References are currently named inline (no `\cite`); add a `.bib` and `\cite` keys for the
  ICAIF'25 Uniswap bridge-swap GNN, ICAIF'24 liquidity-spoofing ABM, ProtoHedge, JaxMARL-HFT,
  and the GENIUS Act / MiCA primary sources.
- Swap `[sigconf,nonacm]` → `[sigconf]` and add the real `\acmConference`/DOI block.
- Every number in the text is reproduced by a committed script; see each repo's `RESULTS.md`.
