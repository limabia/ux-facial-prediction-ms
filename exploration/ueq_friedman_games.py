#!/usr/bin/env python3
"""
Compara as escalas do UEQ-S entre os três jogos com o teste de Friedman.

Os 30 participantes responderam o UEQ-S uma vez por jogo, então as medidas são
pareadas (intra-sujeito) e ordinais: Friedman é o equivalente não-paramétrico da
ANOVA de medidas repetidas. Post-hoc com Wilcoxon pareado + correção de
Bonferroni apenas nas dimensões com resultado significativo.

Lê direto de ``data/raw/ueq/`` (e não dos summaries) porque a análise é sobre o
questionário: roda sem depender do pipeline de frames. Reproduz os números
reportados no data paper (artigo-database/data-paper-jis).

Saídas em exploration/outputs/:
- ueq_friedman_games.csv: chi2, df, p e médias por jogo, para cada dimensão.
- ueq_friedman_posthoc.csv: pares Wilcoxon com p bruto e p corrigido.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy import stats

from modeling.config import DATA_RAW_DIR, PUEQ_GROUPS
from modeling.utils import OUTPUT_DIR, ensure_output_dir

GAMES = ("CatsDrop", "PacMan", "TunelRush")
ALPHA = 0.05


def load_ueq_by_game() -> dict[str, pd.DataFrame]:
    """Uma tabela por jogo, indexada pelo id anônimo do participante."""
    frames = {}
    for game in GAMES:
        path = DATA_RAW_DIR / "ueq" / f"UEQ-{game}-anonimizado.csv"
        if not path.exists():
            path = DATA_RAW_DIR / "ueq" / f"UEQ-{game}.csv"
        if not path.exists():
            raise FileNotFoundError(f"UEQ de {game} não encontrado em {path.parent}")
        frames[game] = pd.read_csv(path, dtype={"idPessoa": str}).set_index("idPessoa")
    return frames


def build_wide(frames: dict[str, pd.DataFrame], items: tuple[str, ...]) -> pd.DataFrame:
    """Participantes nas linhas, jogos nas colunas, valor = média dos itens da escala.

    O ``dropna`` garante o pareamento: só entram participantes com os três jogos.
    """
    return pd.DataFrame(
        {game: frames[game][list(items)].mean(axis=1) for game in GAMES}
    ).dropna()


def run_friedman() -> None:
    print(">>> Friedman: UEQ-S entre jogos...")
    ensure_output_dir()
    frames = load_ueq_by_game()

    rows, posthoc_rows = [], []
    for scale, items in PUEQ_GROUPS.items():
        wide = build_wide(frames, items)
        chi2, p_value = stats.friedmanchisquare(*[wide[game] for game in GAMES])

        row = {
            "escala": scale,
            "n": len(wide),
            "chi2": chi2,
            "df": len(GAMES) - 1,
            "p_value": p_value,
            "significativo": p_value < ALPHA,
        }
        row.update({f"media_{game}": wide[game].mean() for game in GAMES})
        rows.append(row)
        print(
            f"  {scale:15s} n={len(wide)} chi2({len(GAMES) - 1})={chi2:.2f} p={p_value:.3f}"
        )

        if p_value >= ALPHA:
            continue

        # Post-hoc: Friedman é omnibus, indica que há diferença mas não onde.
        pairs = list(combinations(GAMES, 2))
        raw_p = [stats.wilcoxon(wide[a], wide[b]).pvalue for a, b in pairs]
        for (a, b), p_raw in zip(pairs, raw_p, strict=True):
            p_adj = min(p_raw * len(pairs), 1.0)
            posthoc_rows.append(
                {
                    "escala": scale,
                    "jogo_a": a,
                    "jogo_b": b,
                    "p_value": p_raw,
                    "p_bonferroni": p_adj,
                    "significativo": p_adj < ALPHA,
                }
            )
            print(f"      {a} vs {b}: p={p_raw:.4f} (Bonferroni {p_adj:.4f})")

    summary_path = OUTPUT_DIR / "ueq_friedman_games.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSalvo: {summary_path}")

    if posthoc_rows:
        posthoc_path = OUTPUT_DIR / "ueq_friedman_posthoc.csv"
        pd.DataFrame(posthoc_rows).to_csv(posthoc_path, index=False)
        print(f"Salvo: {posthoc_path}")


if __name__ == "__main__":
    run_friedman()
