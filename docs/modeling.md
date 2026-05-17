# Modelagem

- Entrada: `data/interim/frames/summary_max_nwindows20.csv`
  (override via `DATA_INTERIM_DIR`).
- Features de emoção neutra são removidas na etapa preditiva.
- Pré-processamento: z-score por sujeito.

```bash
make analysis              # quality reports + visualizações + grid search
make modeling-refresh      # grid + segmentada (sem geração de plots)
```

## Saídas em `exploration/outputs/`

- `grid_search_experiments.csv` — todas as combinações.
- `model_ci_*.csv` — intervalos de confiança.
- Plots: `plot_error_violin_jogo.png`, `plot_pca_clusters.png`, etc.

Reprodutibilidade: `random_seed=42` em todos os scripts.
