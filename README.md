# Jetson Orin Deploy

Recorte operacional da stack facial para o NVIDIA Jetson Orin Nano. Esta pasta foi isolada para virar um repositorio proprio, separado do fluxo de desenvolvimento do notebook.

## Conteudo

- `compose/`: compose do Jetson.
- `frigate/`: configuracao, cache de modelos e midias persistidas.
- `go2rtc/`: configuracao da camera USB/restream.
- `mosquitto/`: broker MQTT e persistencia local.
- `scripts/`: instalacao, preflight, subida e testes do runtime.
- `vision-app/`: API facial, worker MQTT, embeddings, eventos e modelos.
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
