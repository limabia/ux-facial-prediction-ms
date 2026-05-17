#!/usr/bin/env python3
"""
Realiza análise estatística exploratória dos dados processados.

Objetivos:
1. Comparar distribuições de emoções entre os diferentes jogos (Kruskal-Wallis / ANOVA).
2. Calcular correlações entre emoções médias e as escalas do UEQ (Pearson/Spearman).
3. Gerar visualizações (Boxplots e Heatmaps) para os resultados.

Saídas em exploration/outputs/:
- stats_emotions_by_game.csv: Resultados dos testes de hipótese.
- correlation_emotions_ueq.csv: Matriz de correlação.
- plot_emotions_by_game.png: Boxplot das emoções por jogo.
- plot_correlation_matrix.png: Heatmap das correlações.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from modeling.utils import (
    OUTPUT_DIR,
    UEQ_SCALES,
    ensure_output_dir,
    load_modeling_dataset,
)


def cliffs_delta(x, y):
    """Cliff's Delta — effect size não-paramétrico.

    |d| < 0.147 negligible · 0.147 ≤ |d| < 0.33 small ·
    0.33 ≤ |d| < 0.474 medium · |d| ≥ 0.474 large.
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return np.nan
    dominance = sum(np.sign(xi - yi) for xi in x for yi in y)
    return dominance / (m * n)


def run_statistical_analysis():
    print(">>> Iniciando Análise Estatística Exploratória...")
    ensure_output_dir()

    dataset = load_modeling_dataset(
        summary_name="summary_max_nwindows20.csv",
        target_column="UEQ_Overall",
    )
    df = dataset.dataframe

    emotion_cols = [c for c in df.columns if "_mean_" in c]
    ueq_cols = [c for c in UEQ_SCALES if c in df.columns]

    if "jogo" not in df.columns:
        print("ERRO: Coluna 'jogo' não encontrada no dataframe. Verifique o processamento.")
        return

    print(f"Dados carregados: {len(df)} sessões, {len(emotion_cols)} métricas de emoção.")

    # --- 1. Comparação entre Jogos ---
    print("\n--- 1. Comparando Emoções entre Jogos ---")
    results_games = []
    unique_games = df["jogo"].unique()
    print(f"Jogos encontrados: {unique_games}")

    for emotion in emotion_cols:
        groups = [df[df["jogo"] == g][emotion].dropna() for g in unique_games]
        # Kruskal-Wallis: não-paramétrico (n pequeno, distribuição possivelmente não-normal).
        try:
            stat, p_value = stats.kruskal(*groups)
        except ValueError:
            stat, p_value = float("nan"), float("nan")

        means = {f"mean_{g}": df[df["jogo"] == g][emotion].mean() for g in unique_games}

        row = {
            "metric": emotion,
            "test": "Kruskal-Wallis",
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
        row.update(means)
        results_games.append(row)

    df_games = pd.DataFrame(results_games)

    # Correção FDR (Benjamini-Hochberg) para múltiplas comparações.
    if len(df_games) > 0:
        p_values = df_games["p_value"].values
        rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        df_games["p_value_adjusted"] = p_adjusted
        df_games["significant_adjusted"] = rejected

        print("\nCorreção para múltiplas comparações aplicada (FDR/Benjamini-Hochberg):")
        print(f"  Testes significativos (p < 0.05): {df_games['significant'].sum()}")
        print(f"  Testes significativos após correção: {df_games['significant_adjusted'].sum()}")

    output_games_path = OUTPUT_DIR / "stats_emotions_by_game.csv"
    df_games.sort_values("p_value").to_csv(output_games_path, index=False)
    print(f"Resultados dos testes salvos em: {output_games_path}")

    # Calcular effect sizes (Cliff's Delta) para métricas significativas após correção
    print("\n--- Calculando Effect Sizes (Cliff's Delta) para diferenças significativas ---")
    significant_metrics = df_games[df_games["significant_adjusted"]]["metric"].tolist()

    if significant_metrics:
        effect_sizes = []
        # Limita aos top 10 mais significativos para não inflar comparações par-a-par.
        for emotion in significant_metrics[:10]:
            for i, game1 in enumerate(unique_games):
                for game2 in unique_games[i + 1:]:
                    group1 = df[df["jogo"] == game1][emotion].dropna()
                    group2 = df[df["jogo"] == game2][emotion].dropna()
                    delta = cliffs_delta(group1.values, group2.values)

                    abs_delta = abs(delta)
                    if abs_delta < 0.147:
                        magnitude = "negligible"
                    elif abs_delta < 0.33:
                        magnitude = "small"
                    elif abs_delta < 0.474:
                        magnitude = "medium"
                    else:
                        magnitude = "large"

                    effect_sizes.append({
                        "metric": emotion,
                        "game1": game1,
                        "game2": game2,
                        "cliffs_delta": delta,
                        "magnitude": magnitude
                    })

        df_effects = pd.DataFrame(effect_sizes)
        output_effects_path = OUTPUT_DIR / "stats_emotions_by_game_effects.csv"
        df_effects.to_csv(output_effects_path, index=False)
        print(f"Effect sizes salvos em: {output_effects_path}")

        # Mostrar alguns exemplos
        print("\nExemplos de Effect Sizes (maiores diferenças):")
        top_effects = df_effects.sort_values("cliffs_delta", key=abs, ascending=False).head(5)
        for _, row in top_effects.iterrows():
            print(f"  {row['metric']}: {row['game1']} vs {row['game2']} → δ={row['cliffs_delta']:.3f} ({row['magnitude']})")

    # Plotar Boxplots para as emoções mais significativas (top 4 após correção)
    top_diffs = df_games[df_games["significant_adjusted"]].sort_values("p_value_adjusted").head(4)["metric"].tolist()
    if not top_diffs:
        print("\nNenhuma diferença significativa após correção FDR. Usando top 4 não-corrigidos para visualização.")
        top_diffs = df_games.sort_values("p_value").head(4)["metric"].tolist()
    if top_diffs:
        plt.figure(figsize=(14, 10))
        for i, col in enumerate(top_diffs, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x="jogo", y=col, data=df)
            plt.title(f"{col} (p={df_games[df_games['metric']==col]['p_value'].values[0]:.4f})")
            plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_emotions_by_game.png")
        plt.close()

    # --- 2. Correlação Emoções x UEQ ---
    print("\n--- 2. Calculando Correlações (Emoções vs UEQ) ---")
    if not ueq_cols:
        print("Aviso: Escalas UEQ não encontradas. Pulando correlação.")
    else:
        # Spearman: captura relações monotônicas (não necessariamente lineares).
        corr_data = []
        for ueq in ueq_cols:
            for emotion in emotion_cols:
                valid = df[[ueq, emotion]].dropna()
                if len(valid) < 5:
                    continue
                coef, p_val = stats.spearmanr(valid[ueq], valid[emotion])
                corr_data.append({
                    "ueq_scale": ueq,
                    "emotion_metric": emotion,
                    "spearman_corr": coef,
                    "p_value": p_val,
                    "significant": p_val < 0.05
                })

        df_corr = pd.DataFrame(corr_data)
        output_corr_path = OUTPUT_DIR / "correlation_emotions_ueq.csv"
        df_corr.sort_values("p_value").to_csv(output_corr_path, index=False)
        print(f"Resultados das correlações salvos em: {output_corr_path}")

        pivot_corr = df_corr.pivot(index="emotion_metric", columns="ueq_scale", values="spearman_corr")

        plt.figure(figsize=(10, 12))
        sns.heatmap(pivot_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", cbar_kws={'label': 'Spearman Correlation'})
        plt.title("Correlação: Emoções Faciais vs Experiência (UEQ)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_correlation_matrix.png")
        plt.close()

    print("\n>>> Análise Estatística Concluída.")

if __name__ == "__main__":
    run_statistical_analysis()
