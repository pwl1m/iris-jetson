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

## Modos De Uso Da API

O Iris nao expoe o video original como um servidor RTSP. O worker le a camera
continuamente, atualiza um preview e seleciona frames para inferencia no
intervalo configurado em `STREAM_CAPTURE_INTERVAL_SECONDS`.

- Foto pontual: envie uma imagem para `POST /recognize` ou `POST /compare`.
- Snapshot de evidencia: consulte `GET /captures/latest`, `GET /events` e as
  imagens associadas. Uma captura existe apenas quando um evento foi gravado.
- Preview continuo: use `GET /preview/stream.mjpg` ou o endpoint equivalente
  por camera. E um stream MJPEG de observabilidade; assisti-lo nao aumenta a
  taxa de inferencia.
- Saude e taxa configurada: use `GET /health`, `GET /cameras` e
  `GET /debug/pipeline`. Eles informam o ultimo frame/captura e o intervalo,
  mas nao calculam FPS real de inferencia. Para medir FPS e latencia, execute
  `./scripts/engine_validation.sh` durante uma janela de teste.

Em `GET /health` e `GET /stream/status`, `no_face_detected` conta frames que
foram processados corretamente, mas sem rosto detectavel. Esse caso nao e uma
falha. `recognition_errors` fica reservado para excecoes reais do pipeline.

## Worker De Stream

O worker usa `STREAM_SOURCE_KIND=jetson_gst_usb` como padrao no Jetson, abre cada dispositivo configurado em `CAMERA_n_DEVICE` via OpenCV `CAP_V4L2`, configura FOURCC `MJPG`, resolucao e FPS, processa uma captura por camera a cada `STREAM_CAPTURE_INTERVAL_SECONDS`, salva imagens em `CAPTURE_DIR` e grava eventos em `EVENT_LOG_PATH`.

O nome `jetson_gst_usb` identifica o perfil USB do Jetson, mas o caminho de producao atual e V4L2/OpenCV. O modo opcional `gst_usb_sampled` usa `v4l2src ! jpegparse ! jpegdec ! videorate drop-only=true ! appsink`; ele reduz leituras/copias no Python para a cadencia de captura, mas nao elimina o decode MJPEG anterior ao `videorate`. `nvjpegdec` foi testado com a C930e deste host e falhou na negociacao, portanto nao e usado como fallback automatico.

`STREAM_PREVIEW_ENABLED=false` desliga apenas o preview JPEG/MJPEG da API. As imagens integrais usadas como evidencia e os eventos continuam sendo produzidos.

O pipeline atual separa detector, filtro de qualidade, crop de rosto e reconhecimento `buffalo_m`.

`STREAM_URL`/RTSP via `iris-go2rtc` fica apenas como fallback/observabilidade externa via compose dedicado.

Quando o go2rtc e o dono da USB, `CAMERA_n_STREAM_URL` e o endereço interno
consumido pelo worker (por exemplo, `http://iris-go2rtc:1984/...`).
`CAMERA_n_PUBLIC_STREAM_URL` e o endereço que o endpoint `/cameras` publica
para o Onix/ViewCare abrir no navegador (por exemplo, o IP LAN do Jetson).
Os dois endereços podem, e normalmente devem, ser diferentes.

O campo legado `stream` de `/health` representa a primeira câmera habilitada;
`streams` continua sendo a fonte de status de todas as câmeras. Isso mantém a
saúde do device consistente para o sincronizador Onix quando o slot primário
está desabilitado e outra câmera está em operação.

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

## Integracoes Remotas Opcionais

O reconhecimento e a persistencia local funcionam sem Onix e sem MQTT. As duas
integracoes sao canais de saida independentes e iniciam desligadas:

- `ONIX_PUSH_ENABLED=true` envia cada evento por HTTP POST para
  `ONIX_PUSH_URL`, autenticado com `ONIX_PUSH_TOKEN` e identificado por
  `ONIX_PUSH_DEVICE_UID`.
- `MQTT_PUBLISH_ENABLED=true` publica eventos, oclusoes e snapshots de sujeitos
  no broker definido por `MQTT_PUBLISH_HOST`. Para uso estritamente local, esse
  host pode ser `iris-mosquitto`; para integracao central, deve ser o broker
  central.

As filas de ambos os canais ficam em memoria. Elas evitam bloquear a inferencia,
mas nao constituem uma fila duravel: indisponibilidade prolongada, fila cheia ou
reinicio podem descartar mensagens. A reconciliacao por `GET /events` continua
necessaria quando entrega garantida for requisito.

## Cameras

O runtime processa todas as cameras habilitadas (`CAMERA_1_*` a `CAMERA_4_*`) com um worker independente por camera. A inferencia compartilha o modelo facial sob lock para evitar duplicacao de memoria e concorrencia insegura no Jetson. A API `/cameras` retorna status individual, e previews podem ser consultados em `/cameras/{camera_id}/preview/latest.jpg` ou `/cameras/{camera_id}/preview/stream.mjpg`.

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
