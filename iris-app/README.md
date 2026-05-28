# Iris App

Aplicacao principal da Fase 1: API FastAPI, cadastro facial, reconhecimento e worker de captura direto do stream do `iris-go2rtc`.

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
- `GET /debug/pipeline`
- `GET /cameras`
- `GET /cameras/{camera_id}/status`
- `POST /cameras/{camera_id}/start`
- `POST /cameras/{camera_id}/stop`
- `GET /subjects`
- `DELETE /subjects/{subject}`
- `GET /subjects/{subject}/samples`
- `POST /enroll`
- `POST /recognize`
- `POST /compare`
- `DELETE /samples/{sample_id}`
- `GET /stream/status`
- `POST /stream/start`
- `POST /stream/stop`
- `GET /captures`
- `GET /captures/latest`
- `GET /captures/{capture_id}/image`
- `POST /captures/{capture_id}/enroll?subject=nome`
- `GET /events`
- `GET /events/{event_id}`
- `GET /events/{event_id}/frame.jpg`
- `GET /events/{event_id}/face.jpg`
- `POST /events/{event_id}/enroll?subject=nome`

## Dashboard

- `GET /`: dashboard operacional com captura atual, log de comparacoes, cadastro por captura e gerenciamento de sujeitos/amostras.
- `GET /`: inclui upload manual para cadastro e comparacao, preview da imagem enviada, crop detectado e lista de candidatos.
- `GET /api-help`: referencia simples de endpoints.

## Worker De Stream

O worker consome `STREAM_URL` (padrao: `rtsp://iris-go2rtc:8554/usb_camera`), processa uma captura a cada `STREAM_CAPTURE_INTERVAL_SECONDS`, salva imagens em `CAPTURE_DIR` e grava eventos em `EVENT_LOG_PATH`.

O pipeline atual ja separa detector, filtro de qualidade, crop de rosto e reconhecimento `buffalo_l`, conforme o plano em `../../docs/IRIS_PIPELINE_PLAN_2026-05-29.md`.

No Jetson, o worker tambem pode operar com `STREAM_SOURCE_KIND=jetson_gst_usb`, abrindo a camera USB diretamente por GStreamer com `nvv4l2decoder` e `appsink`.
Nesse modo, `STREAM_URL` deixa de ser a fonte primaria de inferencia e passa a servir apenas como fallback/observabilidade externa.

Cada evento contem:

- `event_id`
- `camera_id`
- `captured_at`
- `frame_image_path`
- `frame_image_url`
- `face_image_path`
- `face_image_url`
- `detector`
- `quality`
- `recognition`

Contrato de integracao atual:

- `GET /events`
- `GET /events/{event_id}`
- `GET /events/{event_id}/frame.jpg`
- `GET /events/{event_id}/face.jpg`
- `POST /events/{event_id}/enroll?subject=nome`
- `GET /debug/pipeline`

## Validacao De Engine

Use `GET /debug/engine` para verificar:

- providers solicitados via `FACE_PROVIDERS`
- providers disponiveis no runtime ONNX
- providers realmente ativos nas sessoes dos modelos InsightFace
- modo atual (`cpu_only` ou `accelerated`)

No Jetson, o `Dockerfile.jetson` instala `onnxruntime-gpu` do indice `pypi.jetson-ai-lab.io` com fallback de providers definido por `FACE_PROVIDERS`.
Padrao recomendado atual no Jetson: `FACE_MODEL_NAME=buffalo_l` e `FACE_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`.
Para usar GPU no InsightFace, configure `FACE_CTX_ID=0` (valor `-1` forca CPU).
Ative `FACE_TRT_FP16=true` e `FACE_TRT_ENGINE_CACHE_PATH=/data/trt-engines` para reaproveitar engines TensorRT entre reinicios.
Antes de subir o stack no Jetson, execute `./scripts/prepare_runtime_libs.sh` para montar cuDNN/TensorRT em `runtime-libs/`.
O `iris-app` agora espera o `iris-go2rtc` responder no probe `STREAM_SOURCE_PROBE_URL` antes de tentar abrir o RTSP.
Quando `STREAM_SOURCE_KIND=jetson_gst_usb`, esse probe e ignorado e o worker abre `/dev/video*` direto.
