# exploration

Scripts headless sobre os summaries. Saídas em `exploration/outputs/`.

| Script | Propósito |
|---|---|
| `statistical_analysis.py` | Kruskal-Wallis + FDR + Cliff's Delta |
| `clustering_analysis.py` | K-Means com Silhouette/Davies-Bouldin/Elbow |
| `demographic_performance_analysis.py` | Viés demográfico + IC bootstrap |
| `compare_faceapi_mediapipe.py` | Séries temporais Face-API vs MediaPipe |
| `generate_quality_reports.py` | Auditoria de qualidade dos summaries |
| `generate_visualizations.py` | Gráficos padrão de emoções por jogo |
| `h2_expressiveness_impact.py` | Hipótese H2 — expressividade facial |
| `h3_events_sanity.py` | Hipótese H3 — features de eventos |
| `h4_multimodal_prediction.py` | Hipótese H4 — predição multimodal |
| `faceapi_sanity_check.py` | Sanity: classificação de jogo (Face-API) |
| `sanity_check_predict_game.py` | Sanity: classificação de jogo (MediaPipe) |

```bash
make analysis              # quality reports + visualizações + grid search
make hypothesis-tests      # H2, H3, H4
make sanity-checks         # sanity checks de classificação
```
