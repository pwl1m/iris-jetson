# Jetson Orin Deploy

Recorte operacional da stack facial para o NVIDIA Jetson Orin Nano. Esta pasta foi isolada para virar um repositorio proprio, separado do fluxo de desenvolvimento do notebook.

## Conteudo

- `compose/`: compose do Jetson.
- `iris-go2rtc/`: configuracao da camera USB/restream.
- `iris-mosquitto/`: broker MQTT e persistencia local.
- `scripts/`: instalacao, preflight, subida e testes do runtime.
- `scripts/prepull_images.sh`: pre-pull manual das imagens para reduzir tempo no primeiro `up`.
- `iris-app/`: API facial, worker de stream, embeddings, eventos e modelos.
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
./scripts/compose_jetson_down.sh
```

## Pre-Pull Manual (Opcional)

```bash
./scripts/prepull_images.sh
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

No fluxo atual, o `iris-app` consome stream do `iris-go2rtc` diretamente e registra eventos/capturas locais para debug operacional.

Existe agora uma variante experimental para Jetson em que o `iris-app` pode consumir a camera USB diretamente por GStreamer/NVIDIA, sem depender do RTSP interno para inferencia.

## Perfil Base Da C930e

Para o Jetson, o baseline atual e capturar a Logitech C930e em `1920x1080` usando `MJPG` na entrada V4L2 e restream RTSP para o stack.
Neste host, `YUYV` em `1080p` limita a camera a `5 fps`, enquanto `MJPG` preserva `1080p` com margem melhor para operacao futura com duas cameras.
