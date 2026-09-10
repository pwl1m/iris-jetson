# Face Models

O `iris-app` usa InsightFace inicialmente.

Modelo configurado no deploy:

- `buffalo_m`

O valor deve permanecer alinhado com `FACE_MODEL_NAME` em `.env.example` e
`docker-compose.yml`. A troca de modelo nao e uma alteracao apenas de
configuracao: antes dela, registrar motivo e impacto na documentacao de
decisoes/plano de modelos definida para o deploy.

O pacote InsightFace baixa modelos para este diretorio quando necessario, se houver rede disponivel.

Para ambiente offline, baixe no notebook, valide, e sincronize este diretorio para o Jetson.
