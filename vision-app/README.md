# Vision App

Aplicacao principal da Fase 1: API FastAPI, cadastro facial, reconhecimento e worker de captura direto do stream do go2rtc.

## Conteudo

- `app/`: codigo da aplicacao.
- `Dockerfile`: imagem para notebook.
- `Dockerfile.jetson`: imagem para Jetson Orin Nano.
- `requirements.txt`: dependencias Python.
- `data-faces/`: banco SQLite de embeddings faciais.
- `data-events/`: auditoria local de reconhecimentos.
- `models/`: modelos do InsightFace/ArcFace.
- `samples/`: imagens e videos para testes offline.

## Endpoints

- `GET /health`
- `GET /debug/engine`
- `GET /subjects`
- `DELETE /subjects/{subject}`
- `GET /subjects/{subject}/samples`
- `POST /enroll`
- `POST /recognize`
- `DELETE /samples/{sample_id}`
- `GET /stream/status`
- `POST /stream/start`
- `POST /stream/stop`
- `GET /captures`
- `GET /captures/latest`
- `GET /captures/{capture_id}/image`
- `POST /captures/{capture_id}/enroll?subject=nome`

## Dashboard

- `GET /`: dashboard operacional com captura atual, log de comparacoes, cadastro por captura e gerenciamento de sujeitos/amostras.
- `GET /api-help`: referencia simples de endpoints.

## Worker De Stream

O worker consome `STREAM_URL` (padrao: `rtsp://go2rtc:8554/usb_camera`), processa uma captura a cada `STREAM_CAPTURE_INTERVAL_SECONDS`, salva imagens em `CAPTURE_DIR` e grava eventos em `EVENT_LOG_PATH`.

Cada evento contem:

- `capture_id`
- `capture_number`
- `camera`
- `captured_at`
- `image_path`
- `image_url`
- `recognition` com status, subject, similarity, face e candidates

## Validacao De Engine

Use `GET /debug/engine` para verificar:

- providers solicitados via `FACE_PROVIDERS`
- providers disponiveis no runtime ONNX
- providers realmente ativos nas sessoes dos modelos InsightFace
- modo atual (`cpu_only` ou `accelerated`)
