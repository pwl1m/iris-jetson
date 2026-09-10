# iris-go2rtc

Entrada e restream da camera USB para observabilidade externa.

O go2rtc usa a fonte V4L2 nativa e preserva o MJPEG da C930e. Como um dispositivo V4L2 so pode ter um dono, este perfil e alternativo ao worker USB direto do `iris-app`: quando o go2rtc for o dono, o Iris deve consumir um stream dele; nunca abra ambos contra a mesma USB.

## Conteudo

- `go2rtc.yaml`: configuracao ativa.
- `go2rtc.yaml.example`: referencia segura sem valores locais.

## Camera

- Notebook: webcam interna via `/dev/video0`.
- Jetson Orin Nano: Logitech C930e via `/dev/video*`.
