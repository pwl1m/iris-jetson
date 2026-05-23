# Vision App

Aplicacao principal da Fase 1: API FastAPI, cadastro facial, reconhecimento e consumo de eventos MQTT do Frigate.

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
- `GET /subjects`
- `POST /enroll`
- `POST /recognize`
