# FaceUX

Pipeline para investigar a relação entre **expressões faciais** e
**experiência do usuário (UEQ-S)** em sessões de jogos digitais.

A partir de vídeos das sessões, extrai frames, infere emoções, consolida em
janelas temporais, cruza com questionários (UEQ-S e PHA/PHD de humor) e
ajusta modelos preditivos (LOSO + grid search com 12 regressores).

## Pergunta de pesquisa

Expressões faciais capturadas durante o uso de software conseguem
**predizer a experiência do usuário** auto-reportada (UEQ-S)?

**Hipóteses testadas:**

- **H1** — Há sinal preditivo de UEQ-S nas emoções faciais agregadas
  (`run_grid_search_experiments.py` cobre ablação emoções vs. PHA vs. controles).
- **H2** — Participantes mais expressivos (maior variabilidade dos
  blendshapes) predizem melhor o contexto de jogo
  (`exploration/h2_expressiveness_impact.py`).
- **H3** — Features baseadas em eventos faciais (razões, máximos) discriminam
  jogos e/ou predizem UEQ-S (`exploration/h3_events_sanity.py`).
- **H4** — Predição multimodal (MediaPipe + PHA + controles) supera cada
  modalidade isolada (`exploration/h4_multimodal_prediction.py`).

> Código liberado como referência metodológica para a dissertação de mestrado
> de **Bianca Lima Santos** (PPgSI/USP). Os vídeos **não são distribuídos**
> (LGPD); a versão anonimizada do dataset (séries de emoções por frame,
> questionários e features por janela) está em [`dataset/`](dataset/), sob
> licença CC BY 4.0. Para coletar dados próprios, siga o formato descrito em
> [_Estrutura de dados_](#estrutura-de-dados).

## Visão geral

Dois caminhos de inferência facial sobre os mesmos vídeos:

| Caminho | Sinal | Stack |
|---|---|---|
| **Face-API.js** | 7 emoções/frame (happy, sad, angry, fearful, disgusted, surprised, neutral) | Node + TF.js |
| **MediaPipe** | 52 blendshapes/frame (Action Units) | Python |

```
                    ┌── Face-API.js ──► JSONs/sessão ──► summaries CSV ─┐
vídeos (data/raw/) ─┤                                                   ├─► modelagem
                    └── MediaPipe   ──► CSVs/sessão  ──► dataset CSV   ─┘
```

## Pré-requisitos

- Python ≥ 3.12 ([uv](https://github.com/astral-sh/uv) recomendado).
- Node.js ≥ 20 (só Face-API).
- Vídeos em `data/raw/*.mov` + questionários em `data/raw/{ueq,humor}/`.
- Pesos do Face-API em `face_expression_recognition/weights/` →
  <https://github.com/justadudewhohacks/face-api.js-models>.
- Modelo do FaceLandmarker em `exploration/face_landmarker.task` →
  <https://developers.google.com/mediapipe/solutions/vision/face_landmarker>.

## Setup

```bash
make setup          # .venv + deps Python
make npm-install    # deps Node (só Face-API)
```

## Pipeline Face-API (principal)

```bash
make pipeline           # extração + inferência + summaries (loop por vídeo)
make modeling-refresh   # grid search LOSO + análise segmentada
make analysis           # quality reports + visualizações + grid search
```

Etapas manuais (caso queira rodar isolado):

```bash
make frames ARGS="data/raw/video.mov"   # extrai frames
make infer                              # Face-API.js sobre os frames
make summaries                          # JSON → CSV + agregação por janelas
```

## Pipeline MediaPipe (alternativo)

```bash
make mediapipe-extract        # blendshapes por vídeo (~3 min/vídeo)
make mediapipe-events         # opcional: features de eventos sobre AUs
make mediapipe-dataset        # consolida em dataset de modelagem
make mediapipe-grid-search    # grid search
```

## Análises extras

```bash
make hypothesis-tests    # H2 (expressividade), H3 (eventos), H4 (multimodal)
make sanity-checks       # classificação de jogo a partir das features
```

> `make analysis`, `make modeling-refresh` e `make mediapipe-*` assumem que
> os summaries/datasets já existem. Sem dados em `data/`, falham.

## Metodologia

- **LOSO** com `sklearn.base.clone()`.
- **12 regressores** × 20 janelas × cenários de ablação.
- **z-score por sujeito**.
- **Kruskal-Wallis + FDR (Benjamini-Hochberg)**, Cliff's Delta.
- **K-Means** validado por Silhouette + Davies-Bouldin.
- Seeds globais (`random_seed=42`).

Detalhes em [`docs/`](docs/).

## Estrutura

```
├── pipeline.py                    # orquestrador Face-API end-to-end
├── frames_extract/                # extração de frames (OpenCV)
├── face_expression_recognition/   # inferência Face-API.js (Node)
├── expressions_csv_exporter/      # JSON → CSV + agregação por janelas
├── features/                      # extração MediaPipe e eventos
├── modeling/                      # grid search, LOSO, segmentada, config
├── exploration/                   # estatística, visualizações, sanity
├── docs/                          # notas metodológicas
├── dataset/                       # dataset público anonimizado (CC BY 4.0)
├── anonymize_data.py              # gera cópias -anonimizado dos questionários
├── export_public_dataset.py       # exporta dataset/ com verificação de PII
├── Makefile · pyproject.toml      # orquestração + deps (uv)
```

## Estrutura de dados

```
data/
├── raw/
│   ├── *.mov                                       # vídeos das sessões
│   ├── controle-experimentos.csv                   # demografia + controles
│   ├── ueq/UEQ-<Jogo>.csv                          # UEQ-S por jogo
│   └── humor/<NN>-QuestionarioHumor-<Etapa>.csv    # PHA/PHD por etapa
├── interim/frames/<sessão>/                        # frames + JSONs por sessão
└── processed/                                      # CSVs consolidados
```

A pasta [`dataset/`](dataset/) segue o mesmo layout (sem vídeos, frames e
JSONs) e pode ser usada no lugar de `data/` via `DATA_RAW_DIR` e
`DATA_INTERIM_DIR`; ver [`dataset/README.md`](dataset/README.md). Para
regerá-la a partir de `data/`: `make public-dataset`.

### Schemas

`controle-experimentos.csv`:

```csv
id,data_realizacao,nome,sexo,idade,formacao_academica,familiaridade_com_jogos,usa_oculos
1,09/02/2024,P001,masculino,34 anos,Graduação,alto,nao
```

`ueq/UEQ-<Jogo>.csv` (escala 1–7, um arquivo por jogo):

```csv
idPessoa,PUEQ1,PUEQ2,PUEQ3,PUEQ4,PUEQ5,PUEQ6,PUEQ7,PUEQ8
1,6,4,7,6,5,7,4,4
```

`humor/<NN>-QuestionarioHumor-<Etapa>.csv` (PHA/PHD por etapa):

```csv
idPessoa,PH1,PH2,PH3,PH4,PH5,PH6,PH7
1,2,6,4,5,6,2,2
```

### Convenções

- Diretórios e nomes de vídeo seguem `<NN>-<Jogo>-<Participante>-<YYYY-MM-DD>`
  (vários scripts fazem `split('-')`).
- Jogos `CatsDrop`, `PacMan`, `TunelRush` aparecem hardcoded em ~10 scripts;
  adaptar a outros conjuntos exige editar essas listas
  (`grep -rn "TunelRush"`).

## Limitações

- N=17 no relatório base; até 30 com coletas adicionais.
- Ambiente controlado, monocâmera — generalização limitada.
- Detecção facial sensível a oclusões e óculos.

## Licença e citação

[MIT](LICENSE) © 2026 Bianca Lima Santos.

```bibtex
@mastersthesis{santos2026faceux,
  author = {Bianca Lima Santos},
  title  = {Análise de Expressões Faciais e Experiência do Usuário em Jogos Digitais},
  school = {PPgSI/USP},
  year   = {2026}
}
```
