#!/usr/bin/env python3
"""Agrega expressions.csv em datasets por janela e enriquece com UEQ/humor/controle.

Para cada sessão em ``data/interim/frames/``, divide o vídeo em N janelas
temporais (N=1..40), agrega cada emoção em cada janela (max, mean, std,
percentis), e produz ``summary_max_nwindowsN.csv``. Em seguida faz merge
com UEQ-S, humor (PHA/PHD) e controles demográficos.

Variáveis de ambiente honradas:
- ``DATA_INTERIM_DIR``: diretório dos expressions.csv (padrão: config.FRAMES_DIR).
- ``DATA_RAW_DIR``: diretório dos questionários (padrão: REPO_ROOT/data/raw).
"""

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from modeling import config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dir(env_var: str, default: Path) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        candidate = Path(env_value).resolve()
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return default


FRAMES_DIR = _resolve_dir("DATA_INTERIM_DIR", config.FRAMES_DIR)
RAW_DIR = _resolve_dir("DATA_RAW_DIR", REPO_ROOT / "data" / "raw")


# Métricas agregadas calculadas para cada emoção em cada janela.
AGGREGATION_FUNCTIONS: dict[str, Any] = {
    "max": lambda series: series.max(),
    "mean": lambda series: series.mean(),
    "std": lambda series: series.std(ddof=0),
    "p25": lambda series: series.quantile(0.25),
    "p50": lambda series: series.quantile(0.50),
    "p75": lambda series: series.quantile(0.75),
    "p90": lambda series: series.quantile(0.90),
    "p95": lambda series: series.quantile(0.95),
}


def find_expression_csvs(frames_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Encontra todos os arquivos expressions.csv nos diretórios de experimentos.
    Extrai ID da pessoa e nome do jogo do nome da pasta do experimento.
    """
    csv_info = []
    experiment_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]

    for exp_dir in experiment_dirs:
        csv_path = exp_dir / "expressions.csv"
        if csv_path.exists():
            experiment_name = exp_dir.name
            id_match = re.match(r'^([^-]+)', experiment_name)
            id_pessoa = id_match.group(1) if id_match else "unknown"

            jogo = "unknown"
            if re.search(r'catsdrop', experiment_name, re.IGNORECASE):
                jogo = "CatsDrop"
            elif re.search(r'pacman', experiment_name, re.IGNORECASE):
                jogo = "PacMan"
            elif re.search(r'tunelrush', experiment_name, re.IGNORECASE):
                jogo = "TunelRush"

            csv_info.append((csv_path, id_pessoa, jogo))

    return csv_info


def normalize_participant_id(value: Any) -> Any | None:
    """Normaliza identificadores para permitir o merge entre fontes distintas."""
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    if value_str.isdigit():
        try:
            return int(value_str)
        except ValueError:
            return value_str

    return value_str.lower()


def load_control_dataset(control_path: Path) -> pd.DataFrame | None:
    """Carrega dados de controle dos participantes (sexo, uso de óculos etc.)."""
    if not control_path.exists():
        print(f"[WARN] Arquivo de controle '{control_path}' não encontrado. Pulando etapa de merge.")
        return None

    try:
        control_df = pd.read_csv(control_path)
    except Exception as exc:
        print(f"[WARN] Falha ao ler '{control_path.name}': {exc}")
        return None

    if 'id' not in control_df.columns:
        print(f"[WARN] Coluna 'id' ausente em '{control_path.name}'. Dados de controle não serão anexados.")
        return None

    control_df = control_df.copy()
    if 'nome' in control_df.columns:
        control_df.drop(columns=['nome'], inplace=True)

    control_df['idPessoa_norm'] = control_df['id'].apply(normalize_participant_id)

    missing_id = control_df['idPessoa_norm'].isna()
    if missing_id.any():
        print(f"[WARN] {missing_id.sum()} registros em '{control_path.name}' sem identificador válido; serão ignorados.")
        control_df = control_df[~missing_id]

    duplicated_ids = control_df['idPessoa_norm'][control_df['idPessoa_norm'].duplicated()].tolist()
    if duplicated_ids:
        print(f"[WARN] IDs duplicados em controle: {duplicated_ids}")

    return control_df


def attach_control_data(summary_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
    """Anexa colunas do dataset de controle aos summaries de expressões."""
    if 'idPessoa' not in summary_df.columns:
        print("[WARN] Coluna 'idPessoa' não encontrada no summary; dados de controle não anexados.")
        return summary_df

    enriched_summary = summary_df.copy()
    enriched_summary['idPessoa_norm'] = enriched_summary['idPessoa'].apply(normalize_participant_id)

    control_columns = [col for col in control_df.columns if col not in {'id', 'idPessoa_norm', 'nome'}]
    merge_basis = control_df[['idPessoa_norm', *control_columns]]

    enriched_summary = enriched_summary.merge(
        merge_basis,
        how='left',
        on='idPessoa_norm'
    )

    enriched_summary.drop(columns=['idPessoa_norm'], inplace=True)

    return enriched_summary


def process_experiment_with_nwindows(
    csv_path: Path,
    id_pessoa: str,
    jogo: str,
    n_windows: int,
    emotion_columns: list[str]
) -> dict[str, Any] | None:
    """
    Processa um arquivo CSV de expressões faciais, dividindo os dados em n janelas
    e calculando métricas agregadas de cada emoção por janela.
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['frame', *emotion_columns]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"  Erro: Colunas ausentes no arquivo CSV {csv_path}: {', '.join(missing_columns)}")
            return None

        total_frames = len(df)
        actual_n_windows = min(n_windows, total_frames)

        exp_result = {
            'idPessoa': id_pessoa,
            'jogo': jogo
        }

        window_size = total_frames / actual_n_windows

        for i in range(actual_n_windows):
            start_idx = int(i * window_size)
            end_idx = int((i + 1) * window_size) if i < actual_n_windows - 1 else total_frames
            window_data = df.iloc[start_idx:end_idx]

            for emotion in emotion_columns:
                emotion_series = window_data[emotion]

                metric_values: dict[str, Any] = {}
                for metric_name, func in AGGREGATION_FUNCTIONS.items():
                    try:
                        metric_values[metric_name] = func(emotion_series)
                    except Exception as agg_error:
                        print(
                            "  Aviso: falha ao calcular a métrica",
                            metric_name,
                            f"para {emotion} na janela {i + 1} do arquivo {csv_path.name}: {agg_error}",
                        )
                        metric_values[metric_name] = None

                # Mantém compatibilidade com o formato anterior utilizando o máximo por janela.
                exp_result[f"{emotion}{i+1}"] = metric_values.get("max")

                for metric_name, value in metric_values.items():
                    column_name = f"{emotion}_{metric_name}_w{i + 1:02d}"
                    exp_result[column_name] = value

        return exp_result

    except Exception as e:
        print(f"  Erro ao processar o experimento {csv_path} com {n_windows} janelas: {e!s}")
        return None


def main():
    frames_dir = FRAMES_DIR
    ueq_dir = RAW_DIR / "ueq"

    if not frames_dir.exists() or not frames_dir.is_dir():
        print(f"Erro: Diretório {frames_dir} não encontrado. Defina DATA_INTERIM_DIR se moveu os dados.")
        return

    csv_info = find_expression_csvs(frames_dir)

    if not csv_info:
        print("Nenhum arquivo expressions.csv encontrado.")
        return

    print(f"Encontrados {len(csv_info)} experimentos")

    emotion_columns = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

    for n_windows in range(1, 41):
        print(f"\nProcessando divisão em {n_windows} janelas...")
        all_results = []

        for csv_path, id_pessoa, jogo in csv_info:
            print(f"  Processando experimento: {csv_path.parent.name}")
            exp_result = process_experiment_with_nwindows(
                csv_path, id_pessoa, jogo, n_windows, emotion_columns
            )
            if exp_result:
                all_results.append(exp_result)

        if all_results:
            results_df = pd.DataFrame(all_results)
            output_path = frames_dir / f"summary_max_nwindows{n_windows}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"  Arquivo consolidado gerado: {output_path}")
        else:
            print(f"  Nenhum resultado para consolidar com {n_windows} janelas")

    print("\nAdicionando dados de UEQ aos arquivos summary...")

    # Caminhos dos arquivos UEQ
    ueq_paths = {
        'TunelRush': ueq_dir / "UEQ-TunelRush.csv",
        'CatsDrop': ueq_dir / "UEQ-CatsDrop.csv",
        'PacMan': ueq_dir / "UEQ-PacMan.csv"
    }

    id_col = "idPessoa"
    game_col = "jogo"
    ueq_columns = [f'PUEQ{i}' for i in range(1, 9)]

    for n_windows in range(1, 41):
        summary_path = frames_dir / f"summary_max_nwindows{n_windows}.csv"

        if not summary_path.exists():
            print(f"[SKIP] Arquivo não encontrado: {summary_path}")
            continue

        print(f"[LOAD] Lendo {summary_path}")
        summary_df = pd.read_csv(summary_path)

        # Inicializa colunas de UEQ com NaN
        for col in ueq_columns:
            summary_df[col] = None

        # Aplica linha a linha o merge apropriado
        for idx, row in summary_df.iterrows():
            jogo = row[game_col]
            id_pessoa = row[id_col]

            ueq_path = ueq_paths.get(jogo)

            if ueq_path is None or not ueq_path.exists():
                print(f"[WARN] Arquivo UEQ para '{jogo}' não encontrado ou inválido.")
                continue

            ueq_df = pd.read_csv(ueq_path)

            # Procura a linha da mesma pessoa
            ueq_row = ueq_df[ueq_df['idPessoa'] == id_pessoa]

            if not ueq_row.empty:
                for col in ueq_columns:
                    summary_df.at[idx, col] = ueq_row.iloc[0][col]
            else:
                print(f"[WARN] Pessoa {id_pessoa} não encontrada no UEQ de {jogo}")

        # Salva de volta o CSV com as colunas UEQ adicionadas
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Atualizado com dados UEQ: {summary_path}")

    print("\nProcessamento concluído!")

    print("\nAdicionando dados de Humor aos arquivos summary...")

    humor_dir = RAW_DIR / "humor"

    humor_map = {
        'TunelRush': {
            'PHA': humor_dir / "01-QuestionarioHumor-inicial.csv",
            'PHD': humor_dir / "02-QuestionarioHumor-tunelCats.csv"
        },
        'CatsDrop': {
            'PHA': humor_dir / "02-QuestionarioHumor-tunelCats.csv",
            'PHD': humor_dir / "03-QuestionarioHumor-catsPacman.csv"
        },
        'PacMan': {
            'PHA': humor_dir / "03-QuestionarioHumor-catsPacman.csv",
            'PHD': humor_dir / "04-QuestionarioHumor-final.csv"
        }
    }

    humor_columns = [f'{prefix}{i}' for prefix in ['PHA', 'PHD'] for i in range(1, 8)]

    for n_windows in range(1, 41):
        summary_path = frames_dir / f"summary_max_nwindows{n_windows}.csv"

        if not summary_path.exists():
            print(f"[SKIP] Arquivo não encontrado: {summary_path}")
            continue

        print(f"[LOAD] Lendo {summary_path}")
        summary_df = pd.read_csv(summary_path)

        # Inicializa colunas de humor com NaN
        for col in humor_columns:
            summary_df[col] = None

        for idx, row in summary_df.iterrows():
            jogo = row[game_col]
            id_pessoa = row[id_col]

            humor_files = humor_map.get(jogo)
            if not humor_files:
                print(f"[WARN] Jogo '{jogo}' não possui mapeamento de humor.")
                continue

            for prefix, file_path in humor_files.items():
                if not file_path.exists():
                    print(f"[WARN] Arquivo de humor '{file_path}' não encontrado.")
                    continue

                humor_df = pd.read_csv(file_path)

                humor_row = humor_df[humor_df['idPessoa'] == id_pessoa]
                if not humor_row.empty:
                    for i in range(1, 8):
                        original_col = f'PH{i}'
                        new_col = f'{prefix}{i}'
                        if original_col in humor_row.columns:
                            summary_df.at[idx, new_col] = humor_row.iloc[0][original_col]
                        else:
                            print(f"[WARN] Coluna {original_col} não encontrada no arquivo {file_path.name}")
                else:
                    print(f"[WARN] Pessoa {id_pessoa} não encontrada em {file_path.name}")

        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Atualizado com dados de humor: {summary_path}")


    control_path = RAW_DIR / "controle-experimentos.csv"
    control_df = load_control_dataset(control_path)

    if control_df is not None:
        print("\nAnexando dados de controle aos arquivos summary...")

        for n_windows in range(1, 41):
            summary_path = frames_dir / f"summary_max_nwindows{n_windows}.csv"

            if not summary_path.exists():
                print(f"[SKIP] Arquivo não encontrado: {summary_path}")
                continue

            summary_df = pd.read_csv(summary_path)
            summary_df = attach_control_data(summary_df, control_df)
            summary_df.to_csv(summary_path, index=False)
            print(f"[OK] Dados de controle anexados: {summary_path}")
    else:
        print("\n[INFO] Dados de controle indisponíveis; etapa de merge ignorada.")

    print("\nOrdenando arquivos summary por idPessoa...")

    for n_windows in range(1, 41):
        summary_path = frames_dir / f"summary_max_nwindows{n_windows}.csv"

        if not summary_path.exists():
            print(f"[SKIP] Arquivo não encontrado: {summary_path}")
            continue

        try:
            df = pd.read_csv(summary_path)

            if "idPessoa" not in df.columns:
                print(f"[WARN] 'idPessoa' não está presente em {summary_path.name}.")
                continue

            df_sorted = df.sort_values(by="idPessoa")
            df_sorted.to_csv(summary_path, index=False)
            print(f"[OK] Arquivo ordenado: {summary_path}")

        except Exception as e:
            print(f"[ERRO] Falha ao ordenar {summary_path.name}: {e!s}")


if __name__ == "__main__":
    main()
