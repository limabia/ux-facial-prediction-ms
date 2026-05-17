# Frames e pré-processamento

`make pipeline` processa cada vídeo em `data/raw/` extraindo frames,
rodando Face-API.js, gerando `expressions.csv` por sessão e apagando
os JPGs ao final. Em seguida dispara `make summaries`.

## Etapas manuais

```bash
make frames ARGS="data/raw/video.mov"
make infer
make summaries
```

## Saídas

- `data/interim/frames/<sessão>/frame_*.jpg` — frames (apagados ao fim do pipeline).
- `data/interim/frames/<sessão>/frame_*.json` — emoções por frame.
- `data/interim/frames/<sessão>/expressions.csv` — consolidado por sessão.
- `data/interim/frames/summary_max_nwindowsN.csv` (N = 1..40) — agregados
  por janela + merge de UEQ-S/humor/controles.

## Notas

- Apenas os primeiros **180 s** de cada vídeo são extraídos.
- Frames sem rosto detectado recebem JSON com expressão neutra padrão.
