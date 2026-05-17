# Notas metodológicas

- **LOSO** com `sklearn.base.clone()` por fold em
  `modeling/run_grid_search_experiments.py`.
- **Seeds globais** (`random_seed=42`) em todos os scripts de modelagem.
- **Múltiplas comparações**: correção Benjamini-Hochberg (FDR) nos testes
  Kruskal-Wallis em `exploration/statistical_analysis.py`.
- **Effect sizes**: Cliff's Delta nas comparações par-a-par entre jogos.
- **Clusterização**: Silhouette, Davies-Bouldin e Elbow em
  `exploration/clustering_analysis.py`.
- **Viés demográfico**: Mann-Whitney/Kruskal-Wallis + IC bootstrap em
  `exploration/demographic_performance_analysis.py`.

## Referências

- Benjamini & Hochberg (1995) — FDR.
- Cliff (1993) — effect size ordinal.
- Rousseeuw (1987) — Silhouette score.
