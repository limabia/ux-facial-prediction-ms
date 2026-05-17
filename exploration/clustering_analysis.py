#!/usr/bin/env python3
"""
Realiza análise de agrupamento (Clustering) não-supervisionado das sessões de jogo.

Objetivos:
1. Identificar perfis comportamentais baseados nas expressões faciais (ex: "Expressivos" vs "Neutros").
2. Verificar se os grupos naturais alinham com o tipo de jogo ou características do usuário.
3. Visualizar a distribuição das sessões em espaço reduzido (PCA).

Saídas em exploration/outputs/:
- clustering_profiles.csv: Centróides dos clusters encontrados.
- plot_pca_clusters.png: Visualização 2D das sessões coloridas por cluster.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from modeling.utils import OUTPUT_DIR, ensure_output_dir, load_modeling_dataset


def run_clustering_analysis():
    print(">>> Iniciando Análise de Clusterização...")
    ensure_output_dir()

    dataset = load_modeling_dataset(target_column="UEQ_Overall", include_controls=False, humor_prefixes=())
    df = dataset.dataframe

    emotion_cols = [c for c in df.columns if "_mean_" in c]
    X = df[emotion_cols].values

    # Normalização crítica para K-Means/PCA.
    X_scaled = StandardScaler().fit_transform(X)

    # 1. PCA para visualização.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    print(f"Variância explicada pelo PCA 2D: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # 2. Validação do número de clusters.
    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []

    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)

        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels_temp))

    # Plotar métricas de validação
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow Method
    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('Número de Clusters (k)')
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True)

    # Silhouette Score (maior é melhor)
    axes[1].plot(k_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('Número de Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis (Maior = Melhor)')
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
    axes[1].grid(True)
    axes[1].legend()

    # Davies-Bouldin Index (menor é melhor)
    axes[2].plot(k_range, davies_bouldin_scores, 'ro-')
    axes[2].set_xlabel('Número de Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin Index (Menor = Melhor)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_cluster_validation.png")
    plt.close()
    print(f"Métricas de validação salvas em: {OUTPUT_DIR / 'plot_cluster_validation.png'}")

    # Determinar k ótimo baseado em Silhouette Score
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)

    print("\nMétricas de validação:")
    print(f"  Melhor k (Silhouette): {best_k_silhouette} (score={best_silhouette:.3f})")
    print(f"  Silhouette scores: {dict(zip(k_range, [f'{s:.3f}' for s in silhouette_scores], strict=False))}")

    # Usar k sugerido ou k=3 se razoável
    optimal_k = best_k_silhouette if best_silhouette > 0.25 else 3
    print(f"\nUsando k={optimal_k} para clusterização final")

    # 3. K-Means com k ótimo + silhouette por amostra.
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    df["Silhouette"] = silhouette_samples(X_scaled, clusters)

    print(f"\nSilhouette Score final (k={optimal_k}): {silhouette_score(X_scaled, clusters):.3f}")

    # 4. Perfil dos clusters (média de emoções por cluster).
    cluster_profile = df.groupby("Cluster")[emotion_cols].mean()
    output_profile = OUTPUT_DIR / "clustering_profiles.csv"
    cluster_profile.to_csv(output_profile)
    print(f"Perfil dos clusters salvo em: {output_profile}")

    # 5. Visualização PCA (cluster e jogo lado a lado).
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=100)
    plt.title("Sessões Agrupadas por Expressão (K-Means)")

    plt.subplot(1, 2, 2)
    if "jogo" in df.columns:
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="jogo", style="jogo", s=100)
        plt.title("Sessões Coloridas por Jogo Real")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_pca_clusters.png")
    plt.close()
    print(f"Gráfico PCA salvo em: {OUTPUT_DIR / 'plot_pca_clusters.png'}")

    # 6. Validação cruzada: diferenças de UEQ entre clusters (Kruskal-Wallis).
    ueq_scales = ["UEQ_Overall", "UEQ_Pragmatic", "UEQ_Hedonic"]
    stats_results = []

    plt.figure(figsize=(15, 5))

    for i, scale in enumerate(ueq_scales, 1):
        if scale not in df.columns:
            print(f"Aviso: Escala {scale} não encontrada.")
            continue

        groups = [df[df["Cluster"] == c][scale].dropna() for c in sorted(df["Cluster"].unique())]
        stat, p_val = stats.kruskal(*groups)

        means = df.groupby("Cluster")[scale].mean().to_dict()
        stats_results.append({
            "ueq_scale": scale,
            "kruskal_stat": stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
            **{f"mean_cluster_{k}": v for k, v in means.items()}
        })

        plt.subplot(1, 3, i)
        sns.boxplot(x="Cluster", y=scale, data=df, palette="viridis")
        plt.title(f"{scale}\np={p_val:.4f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_clusters_vs_ueq.png")
    plt.close()
    print(f"Gráfico Cluster x UEQ salvo em: {OUTPUT_DIR / 'plot_clusters_vs_ueq.png'}")

    # Salvar Stats
    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        output_stats = OUTPUT_DIR / "stats_clusters_vs_ueq.csv"
        df_stats.to_csv(output_stats, index=False)
        print("Resumo Estatístico (Cluster x UEQ):")
        print(df_stats)

    print("\n>>> Análise de Clusterização Concluída.")

if __name__ == "__main__":
    run_clustering_analysis()
