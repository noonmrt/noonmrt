# Coffee Nap — Vídeo 3D Polígonos (80s, 9:16)

Produção completa de vídeo vertical (`1080×1920`, `30 fps`, `80,0 segundos`) em estilo **3D low-poly facetado de alto detalhamento**, com **16 cenários exclusivos alternando a cada 5 segundos exatos**, narração neural em português (`pt-BR-AntonioNeural`), trilha sonora ambiente sintetizada com transições sonoras a cada corte, HUDs científicos animados e legendas cinéticas sincronizadas palavra por palavra.

## Estrutura do Projeto

- `assets/scenes/scene_01.jpg` ... `scene_16.jpg` — 16 cenários 3D em polígonos facetados (9:16)
- `generate_audio.py` — Síntese da narração por cena, extração de timestamps palavra a palavra, trilha sonora procedural e mixagem final (`audio/master_audio.wav` + `build/timeline.json`)
- `render_video.py` — Compositor 1080×1920@30fps com movimento de câmera suave, partículas geométricas 3D, barra de capítulos (16 segmentos), HUDs animados e legendas cinéticas
- `verify_frames.py` — Extrai folha de contato QA para inspeção visual dos frames renderizados
- `output/coffee_nap_3d_poligonos_80s.mp4` — Vídeo final pronto para Reels / TikTok / Shorts

## Reexecutar o Pipeline

```bash
python3 generate_audio.py
python3 render_video.py
python3 verify_frames.py
```
