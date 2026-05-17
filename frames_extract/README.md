# frames_extract

Extrai frames `.jpg` dos vídeos da pasta `data/raw/`, salvando até os primeiros
180 s de cada vídeo em `data/interim/frames/<nome-do-video>/`.

```bash
make frames ARGS="data/raw/video.mov"   # extrai um vídeo
make frames                              # menu curses lista os .mov de data/raw/
```
