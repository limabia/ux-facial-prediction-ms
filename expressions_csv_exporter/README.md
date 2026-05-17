# expressions_csv_exporter

Consolida as saídas do Face-API.js:

- `json_to_csv.py` — varre `data/interim/frames/<sessão>/` e gera um
  `expressions.csv` por sessão (1 linha por frame, 7 colunas de emoção).
- `build_dataset_by_windows.py` — agrega cada `expressions.csv` em
  `summary_max_nwindowsN.csv` (N = 1..40), com merge de UEQ-S, humor e
  controles.

```bash
make summaries     # roda os dois em sequência
```
