# ============================================================
# stablecoin-contagion-gnn  |  Reproducibility Makefile
#
# make all          → full pipeline from raw → paper figures
# make stub         → same but on synthetic data (no API keys needed)
# make test         → unit tests + smoke test
# make clean        → remove generated artefacts (keeps raw data)
# ============================================================

PYTHON  := python
CONFIG  := configs/experiment.yaml
HORIZONS := 30 60 240 1440
BEST_H  := 60
MODEL   := xgboost

.PHONY: all stub dataset train eval interpret clean test lint

# ── Full pipeline (requires data API access) ─────────────────
all: dataset train eval interpret

# ── Stub pipeline (synthetic data, works in CI) ──────────────
stub:
	$(PYTHON) -m scgnn.data.build_dataset --config $(CONFIG) --stub
	@for h in $(HORIZONS); do \
		$(PYTHON) train/run_ladder.py --config $(CONFIG) --horizon $$h --stub; \
	done
	$(PYTHON) eval/run_all_eval.py  --config $(CONFIG) --model $(MODEL)
	$(PYTHON) interpret/hub_report.py  || true
	$(PYTHON) interpret/case_study.py  --config $(CONFIG) --model $(MODEL) || true

# ── Data ─────────────────────────────────────────────────────
dataset: data/processed/dataset_manifest.json

data/processed/dataset_manifest.json: data/episodes.yaml configs/experiment.yaml
	$(PYTHON) -m scgnn.data.build_dataset --config $(CONFIG)

# ── Training (all horizons, all models) ──────────────────────
train: data/processed/dataset_manifest.json
	@for h in $(HORIZONS); do \
		echo "=== Horizon $$h min ==="; \
		$(PYTHON) train/run_ladder.py --config $(CONFIG) --horizon $$h; \
	done

# ── Evaluation ───────────────────────────────────────────────
eval: results/eval/lead_time_$(MODEL).csv

results/eval/lead_time_$(MODEL).csv:
	$(PYTHON) eval/run_all_eval.py --config $(CONFIG) --model $(MODEL)
	$(PYTHON) eval/lead_time_analysis.py --config $(CONFIG) --model $(MODEL)

# ── Interpretability ─────────────────────────────────────────
interpret: results/interpret/hub_report.csv results/interpret/case_study_$(MODEL)_h$(BEST_H).summary.csv

results/interpret/hub_report.csv:
	$(PYTHON) interpret/hub_report.py

results/interpret/case_study_$(MODEL)_h$(BEST_H).summary.csv:
	$(PYTHON) interpret/case_study.py --config $(CONFIG) --model $(MODEL) --horizon $(BEST_H)

# ── Tests ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short
	$(PYTHON) scripts/smoke_test.py

lint:
	ruff check src/ tests/ train/ eval/ interpret/ scripts/

# ── Clean (preserves raw data and cache) ─────────────────────
clean:
	rm -rf data/processed/ results/
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
