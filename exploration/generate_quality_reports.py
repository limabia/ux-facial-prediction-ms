#!/usr/bin/env python3
"""
Gera relatórios de qualidade dos arquivos summary_max_nwindows*.csv.

O script percorre todos os summaries disponíveis em data/interim/frames/,
calcula estatísticas de completude para metadados, UEQ e questionários de
humor e salva relatórios em exploration/outputs/.

Uso:
    python exploration/generate_quality_reports.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from modeling import config

REPO_ROOT = config.REPO_ROOT
FRAMES_DIR = config.FRAMES_DIR
OUTPUT_DIR = config.OUTPUT_DIR

CONTROL_COLUMNS = config.CONTROL_COLUMNS
UEQ_COLUMNS = config.UEQ_COLUMNS
HUMOR_COLUMNS = config.HUMOR_COLUMNS

MissingStats = dict[str, tuple[int, float]]


@dataclass
class SummaryQuality:
    summary_name: str
    rows: int
    participants: int | None
    jogos: int | None
    mean_na_pct: float
    max_na_pct: float
    pct_rows_with_na: float
    missing_control_columns: list[str]
    missing_ueq_columns: list[str]
    missing_humor_columns: list[str]
    warnings: list[str]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_missing(value: str | None) -> bool:
    if value is None:
        return True
    value = str(value).strip()
    return value == "" or value.lower() in {"nan", "none"}


def read_summary(path: Path) -> tuple[Sequence[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    return headers, rows


def compute_missing(
    headers: Iterable[str],
    rows: list[dict[str, str]],
) -> MissingStats:
    total = len(rows)
    stats: MissingStats = {}
    for column in headers:
        missing_count = sum(
            1 for row in rows if column not in row or is_missing(row.get(column))
        )
        missing_pct = (missing_count / total * 100) if total else 0.0
        stats[column] = (missing_count, missing_pct)
    return stats


def write_missing_report(summary_name: str, stats: MissingStats) -> Path:
    output_path = OUTPUT_DIR / f"{summary_name}_missing_values.csv"
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column", "missing_count", "missing_pct"])
        for column, (count, pct) in sorted(
            stats.items(), key=lambda item: item[1][1], reverse=True
        ):
            writer.writerow([column, count, f"{pct:.2f}"])
    return output_path


def evaluate_summary(summary_path: Path) -> SummaryQuality:
    headers, rows = read_summary(summary_path)
    summary_name = summary_path.stem

    missing_stats = compute_missing(headers, rows)
    missing_report_path = write_missing_report(summary_name, missing_stats)

    missing_control = [col for col in CONTROL_COLUMNS if col not in headers]
    missing_ueq = [col for col in UEQ_COLUMNS if col not in headers]
    missing_humor = [col for col in HUMOR_COLUMNS if col not in headers]

    missing_pcts = [stat[1] for stat in missing_stats.values()] or [0.0]
    mean_na_pct = sum(missing_pcts) / len(missing_pcts)
    max_na_pct = max(missing_pcts)

    pct_rows_with_na = (
        sum(1 for row in rows if any(is_missing(row.get(col)) for col in headers))
        / len(rows)
        * 100
        if rows
        else 0.0
    )

    participants = None
    if "idPessoa" in headers:
        participants = len(
            {row["idPessoa"] for row in rows if not is_missing(row.get("idPessoa"))}
        )

    jogos = None
    if "jogo" in headers:
        jogos = len(
            {row["jogo"] for row in rows if not is_missing(row.get("jogo"))}
        )

    warnings: list[str] = []
    if missing_control:
        warnings.append("Colunas de controle ausentes.")
    if missing_ueq:
        warnings.append("Colunas do UEQ ausentes.")
    if missing_humor:
        warnings.append("Colunas de humor ausentes.")
    if pct_rows_with_na > 5:
        warnings.append("Mais de 5% das linhas possuem valores ausentes.")

    quality = SummaryQuality(
        summary_name=summary_path.name,
        rows=len(rows),
        participants=participants,
        jogos=jogos,
        mean_na_pct=round(mean_na_pct, 2),
        max_na_pct=round(max_na_pct, 2),
        pct_rows_with_na=round(pct_rows_with_na, 2),
        missing_control_columns=missing_control,
        missing_ueq_columns=missing_ueq,
        missing_humor_columns=missing_humor,
        warnings=warnings,
    )

    info = asdict(quality)
    info["missing_report_csv"] = str(missing_report_path.relative_to(REPO_ROOT))
    info_path = OUTPUT_DIR / f"{summary_name}_quality.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))

    return quality


def generate_overview_report(qualities: list[SummaryQuality]) -> None:
    if not qualities:
        return

    overview_path = OUTPUT_DIR / "quality_overview.csv"
    fieldnames = list(asdict(qualities[0]).keys())

    with overview_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for quality in qualities:
            writer.writerow(asdict(quality))

    warnings = [
        {"summary_name": q.summary_name, "warnings": q.warnings}
        for q in qualities
        if q.warnings
    ]
    warnings_path = OUTPUT_DIR / "quality_warnings.json"
    warnings_path.write_text(json.dumps(warnings, ensure_ascii=False, indent=2))


def run(selected_summary: str | None = None) -> None:
    ensure_output_dir()

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"Diretório de frames não encontrado: {FRAMES_DIR}. "
            "Execute build_dataset_by_windows.py antes."
        )

    summaries = sorted(FRAMES_DIR.glob("summary_max_nwindows*.csv"))
    if not summaries:
        raise FileNotFoundError(f"Nenhum summary encontrado em {FRAMES_DIR}.")

    if selected_summary:
        summaries = [s for s in summaries if s.name == selected_summary]
        if not summaries:
            raise FileNotFoundError(
                f"O summary '{selected_summary}' não foi encontrado."
            )

    qualities = [evaluate_summary(path) for path in summaries]
    generate_overview_report(qualities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera relatórios de qualidade dos summaries."
    )
    parser.add_argument(
        "--summary",
        help="Nome de um arquivo summary específico (ex.: summary_max_nwindows5.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(selected_summary=args.summary)
