# Gestao de pessoas cadastradas no Iris

## Fonte de verdade e responsabilidades

O Jetson/Iris App e a fonte de verdade para embeddings ArcFace, fotos de
referencia e amostras. O ViewCare deve atuar como interface autenticada e
orquestrador: ele nunca expoe o Iris App diretamente ao navegador.

O marcador de risco e metadado operacional do ViewCare. Ele nao altera o
embedding nem muda a logica de reconhecimento do Jetson.

## Contrato do Iris App

| Operacao | Endpoint Iris | Observacao |
| --- | --- | --- |
| Listar pessoas | `GET /subjects` | Snapshot atual do SQLite do Jetson. |
| Listar fotos | `GET /subjects/{subject}/samples` | Retorna IDs e URLs relativas das amostras. |
| Ver foto | `GET /samples/{sample_id}/image` | Imagem de referencia persistida localmente. |
| Adicionar foto externa | `POST /enroll` | Multipart com `subject` e `file`. |
| Adicionar foto da camera | `POST /cameras/{camera_id}/enroll` | Multipart com `subject`; usa o ultimo frame processado pelo worker Iris. |
| Ver frame para cadastro | `GET /cameras/{camera_id}/enrollment/latest.jpg` | Frame operacional; nao usar preview como fonte biometrica. |
| Renomear | `PATCH /subjects/{subject}` | Multipart com `new_subject`; preserva amostras. |
| Remover uma foto | `DELETE /samples/{sample_id}` | Remove somente a amostra indicada. |
| Remover pessoa | `DELETE /subjects/{subject}` | Operacao destrutiva; deve exigir confirmacao no ViewCare. |

## Fluxo recomendado no ViewCare

1. O operador escolhe o controlador Iris e visualiza o stream ao vivo apenas
   para ajustar angulo, distancia e iluminacao.
2. Para registrar pela camera, o ViewCare solicita a imagem de cadastro ao
   backend e confirma o frame escolhido; o backend chama o endpoint de enroll
   da camera no Iris.
3. Para upload, o ViewCare envia a imagem ao backend; o backend repassa o
   multipart ao `POST /enroll` do device.
4. Depois de criar, renomear, remover ou alterar fotos, o backend solicita um
   novo snapshot de `/subjects` e atualiza o read model por device.
5. O ViewCare grava o marcador de risco e a auditoria de operador. Embeddings
   e imagens permanecem no Jetson.

## Contrato do backend ViewCare

O frontend futuro deve chamar apenas as rotas autenticadas do ViewCare. Elas
validam o vinculo entre pessoa, camera e controlador antes de encaminhar a
operacao ao Iris App.

| Operacao | Endpoint ViewCare | Corpo |
| --- | --- | --- |
| Ver fotos atuais | `GET /api/viewcare/iris/devices/{deviceId}/subjects/{subjectId}/samples` | — |
| Enviar fotos | `POST /api/viewcare/iris/subjects/enroll` | multipart: `device_id`, `subject`, `files[]`, `is_risk_person` |
| Renomear pessoa | `PATCH /api/viewcare/iris/devices/{deviceId}/subjects/{subjectId}` | JSON: `new_subject` |
| Excluir uma foto | `DELETE /api/viewcare/iris/devices/{deviceId}/subjects/{subjectId}/samples/{sampleId}` | — |
| Cadastrar pela camera | `POST /api/viewcare/iris/devices/{deviceId}/cameras/{cameraId}/subjects/enroll` | multipart: `subject`, `is_risk_person` opcional |

Ao renomear, o ViewCare preserva o marcador de risco e marca a associacao
anterior do device como inativa; o historico central nao e apagado. A exclusao
de foto e confirmada no Iris antes de a API responder. O cadastro pela camera
devolve `409` quando nao houver frame recente, situacao que a tela deve
informar ao operador sem tentar reaproveitar o preview.

## Seguranca e operacao

- Fotos faciais e embeddings sao dados biometricos; limitar as acoes de escrita
  a perfis autorizados e registrar operador, data, device e acao.
- Nao usar o MJPEG do go2rtc como imagem de cadastro. Ele e apropriado para
  observacao visual, enquanto o Iris guarda o frame do proprio worker para
  extracao do embedding.
- Nao remover um cadastro inteiro como efeito colateral da exclusao de uma
  foto. A exclusao de pessoa deve ser uma acao separada e confirmada.
- O endpoint de camera retorna `409` enquanto nao existir um frame atual ou o
  stream nao estiver saudavel; ele nunca reutiliza uma imagem antiga depois de
  uma interrupcao da camera.
