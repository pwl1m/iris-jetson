# Vision App

Aplicacao principal da Fase 1: API FastAPI, cadastro facial, reconhecimento e worker de captura direto do stream do go2rtc.

## Conteudo

- `app/`: codigo da aplicacao.
- `Dockerfile.jetson`: imagem para Jetson Orin Nano.
- `requirements.jetson.txt`: dependencias Python para Jetson (ORT instalado separadamente).
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

No Jetson, o worker tambem pode operar com `STREAM_SOURCE_KIND=jetson_gst_usb`, abrindo a camera USB diretamente por GStreamer com `nvv4l2decoder` e `appsink`.
Nesse modo, `STREAM_URL` deixa de ser a fonte primaria de inferencia e passa a servir apenas como fallback/observabilidade externa.

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

No Jetson, o `Dockerfile.jetson` instala `onnxruntime-gpu` do indice `pypi.jetson-ai-lab.io` com fallback de providers definido por `FACE_PROVIDERS`.
Padrao recomendado atual no Jetson: `FACE_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`.
Para usar GPU no InsightFace, configure `FACE_CTX_ID=0` (valor `-1` forca CPU).
Antes de subir o stack no Jetson, execute `./scripts/prepare_runtime_libs.sh` para montar cuDNN/TensorRT em `runtime-libs/`.
O `vision-app` agora espera o `go2rtc` responder no probe `STREAM_SOURCE_PROBE_URL` antes de tentar abrir o RTSP.
Quando `STREAM_SOURCE_KIND=jetson_gst_usb`, esse probe e ignorado e o worker abre `/dev/video*` direto.
