# Iris App

Aplicacao principal da Fase 1: API FastAPI, cadastro facial, reconhecimento e worker de captura direto da camera USB no Jetson.

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
- `GET /captures?limit=20&since=<cursor>`
- `GET /captures/latest`
- `GET /captures/{capture_id}/image`
- `POST /captures/{capture_id}/enroll?subject=nome`
- `GET /events?limit=20&camera_id=entrada&since=<cursor>`
- `GET /events/{event_id}`
- `GET /events/{event_id}/frame.jpg`
- `GET /events/{event_id}/face.jpg`
- `POST /events/{event_id}/enroll?subject=nome`
- `GET /occlusions?limit=50&since=<cursor>&camera_id=entrada&recognition_status=matched&occlusion_class=known_subject_occluded`
- `GET /occlusions/{event_id}`
- `GET /occlusions/{event_id}/frame.jpg`
- `GET /occlusions/{event_id}/face.jpg`

## Dashboard

- `GET /`: dashboard operacional com captura atual, log de comparacoes, cadastro por captura e gerenciamento de sujeitos/amostras.
- `GET /`: inclui upload manual para cadastro e comparacao, preview da imagem enviada, crop detectado e lista de candidatos.
- `GET /api-help`: referencia simples de endpoints.

## Worker De Stream

O worker usa `STREAM_SOURCE_KIND=jetson_gst_usb` como padrao no Jetson, abre `/dev/video0` via OpenCV `CAP_V4L2`, configura FOURCC `MJPG`, resolucao e FPS, processa uma captura a cada `STREAM_CAPTURE_INTERVAL_SECONDS`, salva imagens em `CAPTURE_DIR` e grava eventos em `EVENT_LOG_PATH`.

O pipeline atual ja separa detector, filtro de qualidade, crop de rosto e reconhecimento `buffalo_m`, conforme o plano em `../../docs/IRIS_PIPELINE_PLAN_2026-05-29.md`.

`STREAM_URL`/RTSP via `iris-go2rtc` fica apenas como fallback/observabilidade externa via compose dedicado.

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

O bloco `recognition.face` deve ser tratado como diagnostico operacional do detector. Ele inclui `det_score`, `bbox`, `landmarks_detected`, `total_landmarks`, `landmark_model`, `occluded`, `occlusion_ratio` e `visual_occlusion`. O `new_structure` deve normalizar esses campos em `tb_iris_events` para alimentar o log de comparacoes e auditoria de possivel ocultacao.

## Eventos De Oclusao

Quando o filtro de qualidade classifica `reason=face_ocluida`, o worker grava um evento dedicado em `OCCLUSION_LOG_PATH`. Esse evento nao substitui o reconhecimento normal: ele diferencia criterios operacionais distintos.

O worker tambem pode gravar oclusao com `occlusion_reason=suspected_visual_occlusion`. Esse caso usa uma heuristica leve sobre o crop facial ja detectado, sem modelo extra, para sinalizar capacete, capuz, gorro, pano, papel, sombra forte ou baixa visibilidade de pele quando os landmarks ainda continuam completos.

- `known_subject_occluded`: rosto reconhecido como sujeito cadastrado, mas com oclusao parcial.
- `unknown_subject_occluded`: rosto nao cadastrado ou sem match aceito, tambem com tentativa de oclusao.
- `low_confidence_occlusion`: similaridade perto do limiar, mas insuficiente para decidir com seguranca.

O payload contem `event_id`, `event_type=occlusion`, `occlusion_class`, `captured_at`, `camera_id`, URLs de frame/crop, `detector`, `quality`, `recognition` e o bloco `occlusion` com landmarks, taxa de oclusao, indicador de mudanca subita e landmarks anteriores.

Esse contrato permite ao `new_structure` sincronizar e expor no dominio publico `/api/viewcare/iris/*` tanto o caso "sujeito cadastrado fez oclusao parcial" quanto o caso "rosto desconhecido tentou esconder o rosto".

## Cameras

O MVP atual processa uma camera primaria (`CAMERA_1_ID=entrada`) com um unico worker. A API `/cameras` ja retorna metadados para ate quatro cameras (`CAMERA_1_*` a `CAMERA_4_*`) para que o PHP consiga rastrear futuras configuracoes multi-camera sem mudar o contrato publico.

Enquanto o backend multi-camera nao for implementado, somente a camera primaria retorna `worker_attached=true`; iniciar/parar cameras secundarias retorna conflito operacional.

Contrato de integracao atual:

- `GET /events?limit=20&camera_id=entrada&since=<cursor>`
- `GET /events/{event_id}`
- `GET /events/{event_id}/frame.jpg`
- `GET /events/{event_id}/face.jpg`
- `POST /events/{event_id}/enroll?subject=nome`
- `GET /occlusions?limit=50&since=<cursor>&recognition_status=matched&occlusion_class=known_subject_occluded`
- `GET /occlusions/{event_id}`
- `GET /occlusions/{event_id}/frame.jpg`
- `GET /occlusions/{event_id}/face.jpg`
- `GET /debug/pipeline`

## Validacao De Engine

Use `GET /debug/engine` para verificar:

- providers solicitados via `FACE_PROVIDERS`
- providers disponiveis no runtime ONNX
- providers realmente ativos nas sessoes dos modelos InsightFace
- modo atual (`cpu_only` ou `accelerated`)

No Jetson, o `Dockerfile.jetson` instala `onnxruntime-gpu` do indice `pypi.jetson-ai-lab.io` com fallback de providers definido por `FACE_PROVIDERS`.
Padrao recomendado atual no Jetson: `FACE_MODEL_NAME=buffalo_m` e `FACE_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`.
Para usar GPU no InsightFace, configure `FACE_CTX_ID=0` (valor `-1` forca CPU).
Ative `FACE_TRT_FP16=true` e `FACE_TRT_ENGINE_CACHE_PATH=/data/trt-engines` para reaproveitar engines TensorRT entre reinicios.
Antes de subir o stack no Jetson, execute `./scripts/prepare_runtime_libs.sh` para montar cuDNN/TensorRT em `runtime-libs/`.
Quando `STREAM_SOURCE_KIND=jetson_gst_usb`, esse probe e ignorado e o worker abre `/dev/video*` direto.
