# Jetson Orin Deploy

Recorte operacional da stack facial para o NVIDIA Jetson Orin Nano. Esta pasta foi isolada para virar um repositorio proprio, separado do fluxo de desenvolvimento do notebook.

## Conteudo

- `docker-compose.yml`: compose principal do Jetson na raiz.
- `iris-go2rtc/`: configuracao da camera USB/restream.
- `iris-mosquitto/`: broker MQTT e persistencia local.
- `scripts/`: instalacao, preflight, subida e testes do runtime.
- `scripts/prepull_images.sh`: pre-pull manual das imagens para reduzir tempo no primeiro `up`.
- `iris-app/`: API facial, worker de stream, embeddings, eventos e modelos.
- `docs/ONIX_IRIS_VALIDATION.md`: contrato e validação operacional com Onix/Simtro.
- `.env.example`: contrato de configuracao local.

## Subir No Jetson

```bash
cp .env.example .env
./scripts/install_jetson_dependencies.sh
./scripts/docker_doctor.sh
./scripts/jetson_preflight.sh
./scripts/compose_jetson_up.sh
```

## Parar

```bash
docker compose down
```

## Pre-Pull Manual (Opcional)

```bash
docker compose pull
```

## Deploy Automatizado

O repositorio pode publicar a branch `deploy/staging` no Jetson via GitHub Actions.

Arquivos relevantes:

- `.github/workflows/deploy-jetson-staging.yml`
- `scripts/remote_deploy.sh`

Segredos esperados no GitHub:

- `JETSON_HOST`
- `JETSON_USER`
- `JETSON_SSH_KEY`

Segredos opcionais:

- `JETSON_PORT`
- `JETSON_DEPLOY_PATH`
- `JETSON_HEALTH_URL`

## Escopo

- manter apenas o que for necessario para deploy e operacao no Jetson
- evitar reintroduzir fluxo de notebook aqui
- registrar mudancas de modelo em `../docs/DECISIONS.md` e `../docs/MODEL_PLAN.md`

No fluxo atual, o `iris-go2rtc` e o unico dono V4L2 da USB e expoe cada camera como MJPEG. O `iris-app` consome o MJPEG interno em workers independentes, registra eventos/capturas locais com `camera_id` e publica os transportes configurados para o Onix/ViewCare. Use `CAMERA_n_LAN_STREAM_URL` para a LAN e `CAMERA_n_TAILSCALE_STREAM_URL` para clientes Tailnet; `CAMERA_n_PUBLIC_STREAM_URL` permanece como fallback LAN de compatibilidade.

O modo USB direto por OpenCV/V4L2 e o `gst_usb_sampled` continuam disponiveis para benchmark, mas nao podem disputar a mesma camera com o go2rtc. O modo GStreamer amostrado usa `v4l2src ! jpegdec ! videorate ! appsink`; a C930e ainda precisa ser decodificada antes do `videorate`, portanto ele nao substitui automaticamente o perfil MJPEG atual.

## Perfil Base Da C930e

Para o Jetson, o baseline atual e capturar a Logitech C930e em `1920x1080` usando `MJPG` na entrada V4L2 e disponibilizar MJPEG pelo go2rtc para o Iris e observabilidade.
Neste host, `YUYV` em `1080p` limita a camera a `5 fps`, enquanto `MJPG` preserva `1080p` com margem melhor para operacao futura com duas cameras.
