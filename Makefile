VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
WEIGHTS_DIR ?= face_expression_recognition/weights

.PHONY: setup pipeline frames infer summaries analysis modeling-refresh clean clean-outputs npm-install validate lint \
	mediapipe-extract mediapipe-events mediapipe-dataset mediapipe-grid-search hypothesis-tests sanity-checks

setup:
	@test -d $(VENV) || python -m venv $(VENV)
	@if command -v uv >/dev/null 2>&1; then \
		echo ">> uv encontrado: sincronizando deps"; \
		uv sync; \
		echo ">> instalando projeto como pacote editavel"; \
		uv pip install -e .; \
	else \
		echo ">> uv nao encontrado: instalando via pip"; \
		$(PIP) install --upgrade pip; \
		$(PIP) install -e .; \
	fi

pipeline: ## Executa o processamento completo (Extração -> Inferência -> Limpeza -> Summaries)
	@. $(VENV)/bin/activate && python pipeline.py $(ARGS)

# Alias para compatibilidade ou facilidade
all: pipeline

frames:  ## Extrai frames de um vídeo específico (ARGS="data/raw/video.mov")
	@. $(VENV)/bin/activate && python frames_extract/main.py $(ARGS)

infer: npm-install ## Roda inferencia face-api.js sobre frames
	@cd face_expression_recognition && WEIGHTS_DIR=$(WEIGHTS_DIR) npm run dev

summaries:  ## Converte JSON -> CSV e gera summaries por janela
	@. $(VENV)/bin/activate && python expressions_csv_exporter/json_to_csv.py
	@. $(VENV)/bin/activate && python expressions_csv_exporter/build_dataset_by_windows.py

analysis:  ## Relatorios de qualidade + visualizacoes + grid search (requer summaries gerados)
	@mkdir -p exploration/outputs
	@. $(VENV)/bin/activate && python exploration/generate_quality_reports.py
	@. $(VENV)/bin/activate && python exploration/generate_visualizations.py --summary summary_max_nwindows20.csv
	@. $(VENV)/bin/activate && python modeling/run_grid_search_experiments.py

modeling-refresh:  ## Grid search + segmentada (requer summaries gerados)
	@. $(VENV)/bin/activate && python modeling/run_grid_search_experiments.py
	@. $(VENV)/bin/activate && python modeling/run_segmented_analysis.py

lint:  ## Lint estático (ruff) + bytecode (compileall)
	@. $(VENV)/bin/activate && ruff check
	@. $(VENV)/bin/activate && python -m compileall -q modeling exploration expressions_csv_exporter frames_extract features pipeline.py

npm-install:
	@cd face_expression_recognition && npm ci

validate: lint summaries  ## Smoke test rápido (compile + summaries)

clean-outputs:
	@rm -rf exploration/outputs/__pycache__
	@find exploration/outputs -name "*.tmp" -o -name "*.log" -delete

# === MediaPipe Pipeline (alternativo ao Face-API) ===

mediapipe-extract:  ## Extrai blendshapes de todos os vídeos (lento, ~3min/vídeo)
	@. $(VENV)/bin/activate && python features/extract_mediapipe_full.py

mediapipe-events:  ## Extrai features baseadas em eventos
	@. $(VENV)/bin/activate && python features/extract_events.py

mediapipe-dataset:  ## Gera dataset de modelagem a partir de features MediaPipe
	@. $(VENV)/bin/activate && python modeling/generate_mediapipe_dataset.py

mediapipe-grid-search:  ## Executa grid search com features MediaPipe
	@. $(VENV)/bin/activate && python modeling/run_mediapipe_grid_search.py

# === Testes de Hipótese ===

hypothesis-tests:  ## Executa todos os testes de hipótese (H2, H3, H4)
	@. $(VENV)/bin/activate && python exploration/h2_expressiveness_impact.py
	@. $(VENV)/bin/activate && python exploration/h3_events_sanity.py
	@. $(VENV)/bin/activate && python exploration/h4_multimodal_prediction.py

# === Sanity Checks ===

sanity-checks:  ## Executa sanity checks (classificação de jogo)
	@. $(VENV)/bin/activate && python exploration/sanity_check_predict_game.py
	@. $(VENV)/bin/activate && python exploration/faceapi_sanity_check.py
