# Data Card — stablecoin-contagion-gnn (REAL DATA)

_Built: 2026-06-03T16:17:01.901547Z_

## Provenance

- **Binance** spot 1-minute klines (`api.binance.com/api/v3/klines`), USDT-quoted:
  USDC, TUSD, USDP, FRAX, BUSD, FDUSD, UST.
- **Coinbase** Exchange 1-minute candles (`api.exchange.coinbase.com`), USD-quoted:
  DAI, USDT.
- Binance USDT-quoted prices are converted to **true USD** by multiplying by the
  Coinbase USDT/USD mid (so a USDT depeg is not hidden by the quote currency).
- This is genuine market data, **not** synthetic. Raw pulls cached under `data/raw/`.

## Scope & honest limitations

- The contagion graph is **cross-asset on the deepest venues** (Binance + Coinbase),
  a deliberate scoping choice driven by where reliable 1-min history exists.
- Episodes requiring pre-2019 minute history (e.g. USDT Oct-2018) are **dropped** when
  no venue clears the coverage gate — see `episodes_built` below.
- Single venue per asset means the cross-venue LOP-wedge feature is ~0 (kept for schema
  parity); contagion edges come from cross-asset lead-lag correlation, not LOP.

## Built episodes

| Episode | rows(train/val/test split) |
|---|---|
| USDC_SVB | see base_rates.csv |
| DAI_FTX | see base_rates.csv |
| BUSD_winddown | see base_rates.csv |
| UST_Terra | see base_rates.csv |
| USDT_Oct2018 | see base_rates.csv |
| USDT_May2022 | see base_rates.csv |
| FRAX_SVB | see base_rates.csv |

## Shapes

- feature_dim (F): 48
- n_nodes (N): 9
- train/val/test samples: 2851 / 628 / 845

See `data/processed/base_rates.csv` for positive rates per episode x horizon and
`data/processed/dataset_manifest.json` for the full manifest.