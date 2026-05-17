#!/usr/bin/env python3
"""
Gera visualizações padrão das emoções agregadas e salva gráficos em exploration/outputs/.

Uso:
    python exploration/generate_visualizations.py --summary summary_max_nwindows20.csv
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean, pvariance

import matplotlib.pyplot as plt

from modeling import config

FRAMES_DIR = config.FRAMES_DIR
OUTPUT_DIR = config.OUTPUT_DIR
EMOTIONS = list(config.EMOTIONS)

WINDOW_PATTERN = re.compile(r"_mean_w(\d+)$")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_summary(summary_name: str) -> tuple[list[int], list[dict[str, str]]]:
    path = FRAMES_DIR / summary_name
    if not path.exists():
        raise FileNotFoundError(
            f"O summary '{summary_name}' não foi encontrado em {FRAMES_DIR}."
        )

    rows: list[dict[str, str]] = []
    windows: set[int] = set()

    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        for column in header:
            match = WINDOW_PATTERN.search(column)
            if match:
                windows.add(int(match.group(1)))

        fh.seek(0)
        for line in fh.readlines()[1:]:
            values = line.rstrip("\n").split(",")
            row = dict(zip(header, values, strict=False))
            rows.append(row)

    return sorted(windows), rows


def parse_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def build_long_tables(
    windows: Iterable[int], rows: list[dict[str, str]]
) -> tuple[
    dict[int, dict[str, list[float]]],
    dict[str, dict[int, dict[str, list[float]]]],
    dict[str, dict[str, dict[str, list[float]]]],
]:
    per_window: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_game: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    per_participant: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in rows:
        jogo = row.get("jogo", "unknown")
        participante = row.get("idPessoa", "unknown")

        for window in windows:
            for emotion in EMOTIONS:
                column = f"{emotion}_mean_w{window:02d}"
                value = parse_float(row.get(column, ""))
                if math.isnan(value):
                    continue
                per_window[window][emotion].append(value)
                per_game[jogo][window][emotion].append(value)
                per_participant[jogo][participante][emotion].append(value)

    return per_window, per_game, per_participant


def plot_emotions(
    per_window: dict[int, dict[str, list[float]]], title: str, filename: str
) -> None:
    plt.figure(figsize=(12, 6))
    windows = sorted(per_window.keys())
    for emotion in EMOTIONS:
        means = []
        for window in windows:
            values = per_window[window][emotion]
            means.append(mean(values) if values else math.nan)
        plt.plot(windows, means, marker="o", label=emotion)
    plt.title(title)
    plt.xlabel("Janela")
    plt.ylabel("Intensidade média")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_emotions_by_game(
    per_game: dict[str, dict[int, dict[str, list[float]]]]
) -> None:
    for jogo, window_data in per_game.items():
        filename = f"emotions_{jogo.replace(' ', '_').lower()}.png"
        title = f"Flutuação das emoções — {jogo}"
        plot_emotions(window_data, title, filename)


def plot_variance(
    per_participant: dict[str, dict[str, dict[str, list[float]]]]
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    data: list[list[float]] = []
    labels: list[str] = []
    colors = []
    color_map = {
        "CatsDrop": "#1f77b4",
        "PacMan": "#ff7f0e",
        "TunelRush": "#2ca02c",
    }

    for emotion in EMOTIONS:
        for jogo, participants in per_participant.items():
            variances = [
                pvariance(values)
                for participant_data in participants.values()
                if len(values := participant_data.get(emotion, [])) >= 2
            ]
            if not variances:
                continue
            data.append(variances)
            labels.append(f"{emotion} ({jogo})")
            colors.append(color_map.get(jogo, "#7f7f7f"))

    if not data:
        return

    box = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(box["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("Distribuição da variância por emoção e jogo")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Variância")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "emotions_variance.png", dpi=200)
    plt.close(fig)


def run(summary_name: str) -> None:
    ensure_output_dir()
    windows, rows = parse_summary(summary_name)
    if not windows:
        raise ValueError(
            "Nenhuma coluna *_mean_wXX encontrada no summary. "
            "Execute build_dataset_by_windows.py com métricas de média habilitadas."
        )

    per_window, per_game, per_participant = build_long_tables(windows, rows)
    plot_emotions(per_window, "Flutuação das emoções (todos os jogos)", "emotions_all_games.png")
    plot_emotions_by_game(per_game)
    plot_variance(per_participant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera gráficos padrão das emoções e salva em exploration/outputs."
    )
    parser.add_argument(
        "--summary",
        default="summary_max_nwindows20.csv",
        help="Nome do summary a ser utilizado (padrão: summary_max_nwindows20.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    plt.switch_backend("Agg")
    args = parse_args()
    run(summary_name=args.summary)
