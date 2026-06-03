# ============================================================
# stablecoin-contagion-gnn  |  Reproducibility Makefile
#
# make all          → full pipeline from raw → exports (requires API)
# make stub         → same on synthetic data (no API keys needed)
# make hubs         → hub rankings + export artifacts only
# make figures      → regenerate all paper figures from results/
# make test         → unit tests + smoke test
# make verify       → checksum-verify the data manifest
# make clean        → remove generated artefacts (keeps raw data)
# ============================================================

PYTHON   := python
CONFIG   := configs/experiment.yaml
HORIZONS := 30 60 240 1440
BEST_H   := 60
MODEL    := xgboost

.PHONY: all stub dataset train eval hubs figures interpret verify test lint clean

# ── Full pipeline ─────────────────────────────────────────────
all: dataset train eval hubs figures interpret

# ── Stub pipeline (CI / no-API dev) ──────────────────────────
stub:
	$(PYTHON) -m scgnn.data.build_dataset --config $(CONFIG) --stub
	@for h in $(HORIZONS); do \
		$(PYTHON) train/run_ladder.py --config $(CONFIG) --horizon $$h --stub; \
	done
	$(PYTHON) eval/run_all_eval.py  --config $(CONFIG) --model $(MODEL)
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/export_hubs.py  || true
	$(PYTHON) -m scgnn.export.schema  || true

# ── Data ─────────────────────────────────────────────────────
dataset: data/processed/dataset_manifest.json

data/processed/dataset_manifest.json: data/episodes.yaml configs/experiment.yaml
	$(PYTHON) -m scgnn.data.build_dataset --config $(CONFIG)
	$(PYTHON) -m scgnn.data.manifest build

verify:
	$(PYTHON) -m scgnn.data.manifest verify

# ── Training ─────────────────────────────────────────────────
train: data/processed/dataset_manifest.json
	@for h in $(HORIZONS); do \
		echo "=== Horizon $$h min ==="; \
		$(PYTHON) train/run_ladder.py --config $(CONFIG) --horizon $$h; \
	done

# ── Evaluation ───────────────────────────────────────────────
eval:
	$(PYTHON) eval/run_all_eval.py --config $(CONFIG) --model $(MODEL)
	$(PYTHON) eval/lead_time_analysis.py --config $(CONFIG) --model $(MODEL)

# ── Hub ranking export (critical path) ───────────────────────
hubs: exports/schema_v1.json
	$(PYTHON) scripts/export_hubs.py --config $(CONFIG)

exports/schema_v1.json:
	$(PYTHON) -c "from scgnn.export.schema import write_schema_doc; write_schema_doc()"

# ── Figures ──────────────────────────────────────────────────
figures:
	$(PYTHON) scripts/generate_figures.py

# ── Interpretability ─────────────────────────────────────────
interpret:
	$(PYTHON) interpret/hub_report.py      || true
	$(PYTHON) interpret/case_study.py --config $(CONFIG) --model $(MODEL) || true

# ── Tests ────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short
	$(PYTHON) scripts/smoke_test.py

lint:
	ruff check src/ tests/ train/ eval/ interpret/ scripts/

# ── Clean ────────────────────────────────────────────────────
clean:
	rm -rf data/processed/ results/ exports/*.csv exports/*.json
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
