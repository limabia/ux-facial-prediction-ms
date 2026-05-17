#!/usr/bin/env python3
"""
Analisa o desempenho preditivo (Erro/MAE) segmentado por grupos demográficos e contexto.
Objetivo: Identificar viés no modelo (ex: funciona pior para quem usa óculos? ou para mulheres?).

Segmentações:
1. Sexo (Homem vs Mulher)
2. Uso de Óculos (Sim vs Não)
3. Jogo (PacMan vs CatsDrop vs TunelRush)
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from modeling.utils import (
    OUTPUT_DIR,
    PUEQ_GROUPS,
    UEQ_COLUMNS,
    ensure_output_dir,
    load_modeling_dataset,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PUEQ_ITEMS = list(UEQ_COLUMNS)


def cliffs_delta(x, y):
    """Calcula Cliff's Delta effect size."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return np.nan
    dominance = sum(np.sign(xi - yi) for xi in x for yi in y)
    return dominance / (m * n)


def bootstrap_ci(data, statistic_func=np.mean, n_boot=1000, ci=95):
    """
    Calcula intervalo de confiança bootstrap.

    Args:
        data: array de dados
        statistic_func: função estatística (padrão: média)
        n_boot: número de amostras bootstrap
        ci: nível de confiança (padrão: 95%)

    Returns:
        (lower, upper): limites do intervalo de confiança
    """
    boot_samples = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_samples.append(statistic_func(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_samples, alpha)
    upper = np.percentile(boot_samples, 100 - alpha)
    return lower, upper


MODEL = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

def run_demographic_analysis():
    print(">>> Iniciando Análise Demográfica de Desempenho (Targets: PUEQ1--PUEQ8)...")
    ensure_output_dir()

    results = []

    for target_name in PUEQ_ITEMS:
        # Carregar Dataset COMPLETO (incluindo colunas de controle para segmentação)
        dataset = load_modeling_dataset(
            target_column=target_name,
            emotion_metrics=("mean",),
            humor_prefixes=(), # Sem humor, apenas Face
            include_controls=True
        )

        df = dataset.dataframe
        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        print(f"Executando LOSO para {target_name}...")
        logo = LeaveOneGroupOut()

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Escalonamento
            scaler_fold = StandardScaler()
            X_train_s = scaler_fold.fit_transform(X_train)
            X_test_s = scaler_fold.transform(X_test)

            # Clonar modelo para garantir independência entre folds
            model = clone(MODEL)

            # Treinar
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            # Guardar resultados junto com metadados das instâncias de teste
            for i, idx_original in enumerate(test_idx):
                row_data = df.iloc[idx_original]
                results.append({
                    "target": target_name,
                    "idPessoa": row_data.get("idPessoa", groups[test_idx][i]),
                    "y_true": y_test[i],
                    "y_pred": preds[i],
                    "error_abs": abs(y_test[i] - preds[i]),
                    "sexo": row_data.get("sexo"),
                    "usa_oculos": row_data.get("usa_oculos"),
                    "jogo": row_data.get("jogo")
                })

    df_res = pd.DataFrame(results)

    # Salvar dados por item
    items_path = OUTPUT_DIR / "demographic_performance_predictions_items.csv"
    df_res.to_csv(items_path, index=False)

    # Agregar erro por escala (média dos itens)
    agg_rows = []
    group_cols = ["idPessoa", "sexo", "usa_oculos", "jogo"]
    for scale_name, items in PUEQ_GROUPS.items():
        subset = df_res[df_res["target"].isin(items)].copy()
        if subset.empty:
            continue
        grouped = subset.groupby(group_cols, dropna=False)["error_abs"].mean().reset_index()
        grouped["target"] = scale_name
        agg_rows.append(grouped)

    df_agg = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()

    # --- Análise e Plotagem (agregado por escala) ---
    for scale_name in PUEQ_GROUPS:
        scale_df = df_agg[df_agg["target"] == scale_name]
        if scale_df.empty:
            continue
        print(f"\n### {scale_name} ###")
        analyze_segment(scale_df, "sexo", "Performance por Sexo")
        analyze_segment(scale_df, "usa_oculos", "Performance por Uso de Óculos")
        analyze_segment(scale_df, "jogo", "Performance por Jogo")

    # Salvar dados agregados
    df_agg.to_csv(OUTPUT_DIR / "demographic_performance_predictions.csv", index=False)
    print("\n>>> Análise Demográfica Concluída.")

def analyze_segment(df, column, title):
    if column not in df.columns:
        print(f"Aviso: Coluna {column} não encontrada.")
        return

    print(f"\n--- {title} ---")

    # Agrupar e calcular estatísticas com intervalos de confiança
    groups = df.groupby(column)["error_abs"]

    summary_data = []
    for group_name, group_data in groups:
        errors = group_data.values
        mean_error = np.mean(errors)

        # Bootstrap CI para MAE
        ci_lower, ci_upper = bootstrap_ci(errors, statistic_func=np.mean, n_boot=1000)

        summary_data.append({
            column: group_name,
            "MAE": mean_error,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "Count": len(errors),
            "Std": np.std(errors),
            "Median": np.median(errors),
            "IQR": np.percentile(errors, 75) - np.percentile(errors, 25)
        })

    summary = pd.DataFrame(summary_data)

    if len(summary) == 0:
        print(f"  AVISO: Nenhum dado encontrado para segmentação por '{column}'")
        return

    print(summary)

    # Testes estatísticos se houver 2+ grupos
    unique_groups = df[column].unique()
    if len(unique_groups) >= 2:
        print("\n  Testes Estatísticos:")

        # Teste de Mann-Whitney (para 2 grupos) ou Kruskal-Wallis (para 3+)
        group_errors = [df[df[column] == g]["error_abs"].values for g in unique_groups]

        if len(unique_groups) == 2:
            stat, p_value = scipy_stats.mannwhitneyu(group_errors[0], group_errors[1], alternative='two-sided')
            test_name = "Mann-Whitney U"

            # Effect size (Cliff's Delta)
            delta = cliffs_delta(group_errors[0], group_errors[1])
            abs_delta = abs(delta)
            if abs_delta < 0.147:
                magnitude = "negligible"
            elif abs_delta < 0.33:
                magnitude = "small"
            elif abs_delta < 0.474:
                magnitude = "medium"
            else:
                magnitude = "large"

            print(f"    {test_name}: statistic={stat:.3f}, p-value={p_value:.4f}")
            print(f"    Effect Size (Cliff's Delta): δ={delta:.3f} ({magnitude})")

        else:
            stat, p_value = scipy_stats.kruskal(*group_errors)
            test_name = "Kruskal-Wallis"
            print(f"    {test_name}: H={stat:.3f}, p-value={p_value:.4f}")

            # Post-hoc pairwise comparisons se significativo
            if p_value < 0.05:
                print("    Post-hoc (pairwise Mann-Whitney com Bonferroni):")
                n_comparisons = len(unique_groups) * (len(unique_groups) - 1) // 2
                alpha_corrected = 0.05 / n_comparisons

                for i, group1 in enumerate(unique_groups):
                    for group2 in unique_groups[i+1:]:
                        g1_errors = df[df[column] == group1]["error_abs"].values
                        g2_errors = df[df[column] == group2]["error_abs"].values
                        _, p_pair = scipy_stats.mannwhitneyu(g1_errors, g2_errors)
                        delta_pair = cliffs_delta(g1_errors, g2_errors)

                        sig = "***" if p_pair < alpha_corrected else ""
                        print(f"      {group1} vs {group2}: p={p_pair:.4f}{sig}, δ={delta_pair:.3f}")

        summary["p_value"] = p_value
        summary["test"] = test_name

    summary.to_csv(OUTPUT_DIR / f"stats_performance_{column}.csv", index=False)

    # Plot com error bars (IC 95%)
    plt.figure(figsize=(10, 6))

    x_pos = range(len(summary))
    plt.bar(x_pos, summary["MAE"], yerr=[
        summary["MAE"] - summary["CI_Lower"],
        summary["CI_Upper"] - summary["MAE"]
    ], capsize=10, alpha=0.7, color='teal', edgecolor='black')

    plt.xticks(x_pos, summary[column])
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Erro Médio Absoluto (MAE)")
    plt.title(f"{title} - MAE com IC 95% Bootstrap\n(Quanto menor, melhor)")
    plt.grid(axis='y', alpha=0.3)

    # Adicionar labels de valor com IC
    for i, row in summary.iterrows():
        plt.text(i, row["CI_Upper"] + 0.05,
                f"{row['MAE']:.2f}\n[{row['CI_Lower']:.2f}, {row['CI_Upper']:.2f}]",
                ha="center", fontsize=9)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"plot_performance_{column}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")

if __name__ == "__main__":
    run_demographic_analysis()
