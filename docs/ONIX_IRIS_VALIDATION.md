# Validação Onix ↔ Iris

Data da validação: 2026-09-11 (America/Sao_Paulo).

## Topologia em operação

```text
C930e USB -> go2rtc (único dono V4L2) -> MJPEG público
                                         -> iris-app (MJPEG interno)
iris-app -> API HTTP / MQTT -> sincronização Iris no Onix -> /api/viewcare/iris/* -> Simtro
                                                                  |                     |
                                                           resolve LAN/Tailnet      MJPEG direto
```

O `CAMERA_n_STREAM_URL` é usado internamente pelo `iris-app`, via DNS Docker
`iris-go2rtc`. Para visualização, o Iris publica `stream_urls` no inventário:
`CAMERA_n_LAN_STREAM_URL` para a LAN e
`CAMERA_n_TAILSCALE_STREAM_URL` para a Tailnet. O legado
`CAMERA_n_PUBLIC_STREAM_URL` permanece como fallback LAN durante a migração.
Não usar a URL interna como URL pública.

## Resultado da validação autenticada

O teste foi realizado por login de homologação, usando o JWT apenas em arquivo
temporário durante a chamada. Credenciais, JWTs e cookies não pertencem a este
repositório.

| Contrato Onix | Resultado |
|---|---|
| `GET /api/viewcare/iris/dashboard/summary` | 200; 1 device online, 0 degradados, 1 câmera online, `sync_state=ok`. |
| `GET /api/viewcare/iris/devices` | 200; device com `status=online`, `health_status=ok` e último sync bem-sucedido. |
| `GET /api/viewcare/iris/cameras` | 200; `entrada_2` online, MJPEG, 1920x1080 a 15 FPS, URL pública LAN e worker anexado. |
| `GET /api/viewcare/iris/cameras/{id}/stream` | 200; resolvedor autenticado para abrir a câmera, retornando `stream_url`, `render_mode=mjpeg`, disponibilidade e metadados. |
| `GET /api/viewcare/iris/debug/health` | 200; controlador e sincronização saudáveis. |
| `GET /api/viewcare/iris/debug/sync-runs` | 200; último ciclo concluído com sucesso. |
| Eventos, atividades, oclusões e alertas | 200; read model populado. |
| `GET /api/viewcare/iris/events/{id}/frame` | 200, JPEG validado. |
| `GET /api/viewcare/iris/events/{id}/face` | 200, JPEG validado. |

O servidor Onix também alcançou diretamente o Jetson em `/health`, `/cameras`
e no snapshot público do go2rtc. A câmera respondeu `active=true`, sem erro,
com 1920x1080 MJPEG a 15 FPS.

## Saúde do device

O campo legado `health.stream` precisa apontar para uma câmera habilitada. O
sincronizador Onix o combina com `health.streams`: se o resumo apontar para um
slot primário desabilitado, o device fica `degraded` mesmo com uma câmera ativa.
O Iris seleciona agora a primeira câmera habilitada para esse resumo e mantém
todas as câmeras em `streams`.

## Stream no Simtro

O backend Onix persiste os transportes publicados pelo Iris, mas entrega ao
browser somente um `stream_url` já resolvido. O componente **Câmera ao vivo**
do SIMTRO usa essa URL como `src` do MJPEG; ele não monta URLs do Jetson, não
expõe a lista de transportes e não retransmite vídeo pelo Onix.

### Qualidade, desempenho e banda

O endpoint de abertura retorna a mesma URL MJPEG do go2rtc. Não existe uma
segunda codificação no Onix, portanto a qualidade, resolução e FPS são os
mesmos observados diretamente no Jetson: 1920x1080 a aproximadamente 15 FPS.
Em uma medição de 10 segundos, um cliente recebeu 45,9 MB, aproximadamente
36,7 Mbps. Esse valor varia conforme a cena e a compressão MJPEG.

O consumo observado no Jetson foi de aproximadamente 2% de CPU no go2rtc sem
cliente e 2,8% com um cliente, com cerca de 11,8 MiB de memória. O Iris
continua sendo o maior consumidor por causa da inferência facial. Como o
navegador acessa o Jetson diretamente, o Onix não processa nem retransmite o
vídeo; cada cliente adicional acrescenta banda de saída no Jetson, mas não
cria uma cópia de inferência no Iris.

### Endpoint de abertura/renderização

`GET /api/viewcare/iris/cameras/{id}/stream` usa a autenticação/permissão
normal do ViewCare e retorna somente o contrato de abertura. Para a câmera 18,
o campo `stream_url` é o MJPEG público do go2rtc. O front pode utilizá-lo como
`src` de um elemento `<img>` ou abrir em nova aba; o PHP não retransmite os
frames. `available=false` sinaliza que a câmera está cadastrada, mas sem URL
ou sem estado online.

### Seleção automática de rede

O SIMTRO determina o contexto sem apresentar escolha ao operador: quando ele é
aberto por um IP Tailscale (`100.64.0.0/10`), chama o endpoint com
`network=tailnet`; quando é aberto por uma faixa privada RFC1918, usa
`network=lan`. O Onix então devolve exclusivamente a URL correspondente. O
mapa interno `stream_urls` nunca integra a resposta consumida pelo front.

Esse mecanismo não torna uma URL LAN roteável pela Tailnet: ele entrega o
transporte que o navegador já consegue alcançar. Hostnames futuros precisam
de um contexto de rede explícito no mesmo resolvedor, sem criar seletor na UI.

Exemplo de resposta (campos estáveis):

```json
{
  "data": {
    "camera_id": 18,
    "camera_key": "entrada_2",
    "name": "ENTRADA 2",
    "stream_url": "http://<jetson-tailnet-ou-lan>:1984/api/stream.mjpeg?src=usb_camera_2",
    "render_mode": "mjpeg",
    "content_type": "multipart/x-mixed-replace",
    "available": true,
    "status": "online",
    "input_format": "mjpeg",
    "resolution": {"width": 1920, "height": 1080, "fps": 15}
  }
}
```

## Segunda câmera

Uma segunda câmera adiciona outra câmera ao mesmo device Iris, não outro
controlador. O fluxo é:

1. Conectar a câmera em controlador USB apropriado, identificar seu serial e
   aprová-lo no runtime.
2. Criar `usb_camera_3` no go2rtc, com mapeamento V4L2 exclusivo para ela.
3. Configurar `CAMERA_3_*`: `ENABLED=true`, fonte `http_mjpeg`, URL interna,
   `CAMERA_3_LAN_STREAM_URL` e, quando aplicável,
   `CAMERA_3_TAILSCALE_STREAM_URL`.
4. O Iris inicia um worker próprio e grava eventos com `camera_id=entrada_3`.
5. O próximo sync cria/atualiza a segunda linha de câmera no Onix; os eventos
   continuam separados por `camera_id` e serial.

O go2rtc encaminha MJPEG da C930e com baixo consumo de CPU, mas a inferência
facial é compartilhada pelo Iris. Validar duas câmeras antes de habilitar uma
terceira. Preferir controlador USB distinto: a C930e atual está em um hub USB
2.0 de 480 Mbps.

## Modos de ingestão

O push HTTP do Iris é opcional. Com `ONIX_PUSH_ENABLED=false`, o read model é
atualizado pelo sync HTTP agendado no Onix a cada dois minutos. A ativação de
push ou MQTT não altera o contrato público do Simtro; apenas reduz a latência
de materialização.
