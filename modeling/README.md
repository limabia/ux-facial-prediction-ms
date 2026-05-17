# modeling

Experimentos preditivos com validação Leave-One-Subject-Out (LOSO).

- **X**: emoções agregadas por janela (+ opcional PHA + controles).
- **Y**: cada `PUEQ1..PUEQ8` (escala 1–7).
- Pré-processamento: z-score por sujeito.

## Scripts

- `run_grid_search_experiments.py` — grid principal (12 regressores ×
  20 janelas × 4 cenários de ablação). Gera
  `exploration/outputs/grid_search_experiments.csv`.
- `run_segmented_analysis.py` — segmenta por gênero e jogo no melhor cenário.
- `generate_mediapipe_dataset.py` / `run_mediapipe_grid_search.py` —
  variante MediaPipe.
- `utils.py` — `load_modeling_dataset()`, anti-leakage, z-score.
- `config.py` — caminhos e constantes do projeto (`EMOTIONS`, `UEQ_COLUMNS`,
  `PUEQ_GROUPS`, etc.).

```bash
make modeling-refresh        # grid + segmentada
make mediapipe-grid-search   # variante MediaPipe
```
