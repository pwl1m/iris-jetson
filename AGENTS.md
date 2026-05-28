# AGENTS.md

## Objetivo

Esta pasta representa o futuro repositorio de deploy do Jetson Orin Nano. O foco aqui e runtime, operacao e manutencao do pipeline facial em borda.

## Escopo

- `docker-compose.yml`: compose principal do Jetson na raiz do deploy.
- `iris-app/`: API facial, worker MQTT, SQLite e modelos faciais.
- `frigate/`, `iris-go2rtc/`, `iris-mosquitto/`: componentes de runtime e persistencia local.
- `scripts/`: instalacao, preflight, compose e testes operacionais.

## Regras

- Nao reintroduzir artefatos especificos de notebook nesta pasta.
- Preservar compatibilidade com `linux/arm64` e JetPack 6.x.
- Manter scripts idempotentes sempre que possivel.
- Validar camera, Docker, runtime NVIDIA e recursos do host antes de subir containers.
- Nao commit credenciais reais. Use `.env.example` como contrato.
- O deploy automatizado deve sair da branch `deploy/staging` deste repositorio.

## Mudancas De Modelo

- Antes de trocar modelo facial ou detector, registrar motivo e impacto em `../docs/DECISIONS.md`.
- Toda alteracao que afete a estrategia de modelos deve refletir `../docs/MODEL_PLAN.md`.
