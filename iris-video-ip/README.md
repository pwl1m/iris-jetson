# Iris video IP — primeira implementação

Aquisição H.265 por RTSP/TCP com `nvv4l2decoder` e entrega autenticada ao Iris.
O transporte padrão é NV12: a engine mantém o frame nesse formato após o decode,
sem conversão contínua para BGR. O `iris-app` converte somente o frame amostrado
que será analisado. A baseline foi validada com stream H.265 real em 18/09/2026.
Não contém decisões de identidade.

Após o decoder, `videorate drop-only` limita a saída para `IRIS_IP_OUTPUT_FPS=10`
por padrão. O decoder continua recebendo e decodificando o fluxo H.265, mas
`nvvidconv`, o empacotamento NV12 e o transporte HTTP não são executados para
cada frame de 20 FPS.

`retinaface_decode.py` contém o decoder de referência do detector `buffalo_m`.
`retinaface_parser.cpp` é o parser C++ experimental correspondente. Ambos ainda
não são carregados pela engine de produção: a referência foi validada contra o
InsightFace e o parser já passou pelo carregamento/smoke test do `gst-nvinfer`,
mas o shadow mode ainda não tem paridade aprovada (o detector alternou entre
zero e dois objetos numa cena sem rosto). O arquivo
`retinaface_nvinfer.txt` contém somente esse config experimental e não é usado
no Compose IP.

## Contrato v2 (NV12; BGR v1 como fallback)

- `GET /health`: 200 somente quando existe frame decodificado recente, senão 503.
- `GET /v1/frame`: exige `Authorization: Bearer <token>`; 401 sem autenticação,
  503 sem frame recente; 200 com NV12 compacto por padrão e `X-Iris-Frame` JSON.
- Metadados: versão, câmera, epoch de reconexão, sequência, largura, altura,
  formato e instante monotônico de disponibilização do frame decodificado.
  `version=2, format=NV12` é o caminho padrão; `version=1, format=BGR` continua
  disponível como fallback compatível.
- `captured_at=null`: não inventar horário de captura na câmera. A idade mede
  disponibilidade local, não latência completa sensor→aplicação.
- Uma única posição de frame, substituída continuamente; appsink e fila limitados.
  O cliente rejeita tamanho inválido, frame expirado, duplicado ou fora de ordem.
- Limite de 32 MiB por frame. Containers no mesmo Jetson/relógio monotônico.
  Contrato não serve a hosts remotos sem revisão de timestamp e TLS.

O transporte HTTP fica somente na rede Docker do projeto e usa token de pelo
menos 32 caracteres ASCII sem espaços. Token e URI vêm de arquivos montados como
secrets; nunca de URL pública, log ou variável contendo a credencial. A porta
8090 não é publicada diretamente: a porta opcional 1986 expõe somente o preview
JPEG/MJPEG, sem acesso ao contrato `/v1/frame`.

## Limites intencionais desta etapa

Há uma cópia compacta do frame decodificado para o transporte NV12 (cerca de 3,1
MB por frame 1080p). Isso elimina a conversão contínua para BGR e reduz pela metade
o payload em relação à baseline anterior. O cliente amostra em 10 Hz por padrão;
os modelos e regras existentes permanecem no app. Ainda não é zero-copy nem a
arquitetura final de inferência NVIDIA. Start/stop de câmera controla
o consumidor Iris, não desliga a aquisição independente da engine.

O preview público é produzido pela própria engine depois do mesmo decode NVIDIA:
`GET /v1/preview.jpg` entrega a última imagem e `GET /v1/preview.mjpeg` entrega
MJPEG. O ramo é independente, com JPEG a 5 FPS por padrão e `nvjpegenc`; o
go2rtc permanece opcional e desligado na linha IP, pois não converteu o H.265
da câmera para MJPEG. O contrato e a configuração LAN/Onix estão em
[`docs/ip-camera/10_ENGINE_MJPEG_PREVIEW.md`](../docs/ip-camera/10_ENGINE_MJPEG_PREVIEW.md).

## Validação sem hardware

Na raiz do repositório:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/ip_camera -v
docker compose -p iris-ip -f docker-compose.ip-camera.yml config --quiet
```

Os testes usam apenas frames sintéticos. Nenhuma câmera ou container é acionado.

## Preparação do ensaio real — ainda pendente

1. Conferir versão JetPack/L4T, imagem arm64 e dependências NVIDIA; fixar digest.
2. Registrar saúde Iris e recursos antes do build/ensaio. Não iniciar carga se
   o Iris já estiver sem saúde; seguir os limites de `docs/ip-camera`.
3. Provisionar fora do Git os arquivos de URI e token (modo 0600) e, somente para
   o profile preview, o YAML privado do go2rtc. Usar `.env.example` para os caminhos.
4. Construir a imagem IP e verificar plugins dentro dela antes de ler a câmera.
5. Executar primeiro apenas a engine, sem modelos, e confirmar frames/reconexão.
6. Validar o app IP em dados isolados; sem trocar modelo, thresholds ou fontes USB.

Nenhum secret real acompanha este código. Não montar diretório privado de runtime
de outro projeto e não alterar pacotes globais do host.

Referências: [NVIDIA Accelerated GStreamer](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
e [appsink](https://gstreamer.freedesktop.org/documentation/app/appsink.html).
