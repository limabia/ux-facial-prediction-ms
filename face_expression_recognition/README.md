# face_expression_recognition

Módulo Node que aplica [`face-api.js`](https://github.com/justadudewhohacks/face-api.js)
(via [`@vladmandic/face-api`](https://github.com/vladmandic/face-api)) sobre
TensorFlow.js a cada frame `.jpg` em `data/interim/frames/<sessão>/` e salva,
ao lado, um `.json` com as 7 probabilidades de emoção.

## Setup

- Node.js ≥ 20.
- `npm ci`
- Pesos em `./weights/` — baixar de
  <https://github.com/justadudewhohacks/face-api.js-models>.

## Como rodar

Via Makefile da raiz:

```bash
make infer
```

Equivale a `cd face_expression_recognition && WEIGHTS_DIR=./weights npm run dev`.

Variável `SINGLE_FOLDER` (opcional) processa apenas uma sessão; sem ela,
itera por todas as subpastas de `data/interim/frames/`.

## Regras

- Apenas `.png`/`.jpg`/`.jpeg` são processados; detecções com confiança < 0.5
  são descartadas.
- Sem rosto detectado ou erro → JSON com expressão neutra padrão.
