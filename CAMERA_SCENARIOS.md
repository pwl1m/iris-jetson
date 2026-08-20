# Cenários de Câmera — Iris Jetson Orin Nano

Data: 2026-06-01

## Hardware Base

| Item | Especificação |
|---|---|
| **Placa** | NVIDIA Jetson Orin Nano Dev Kit Super (8 GB) |
| **CPU** | 6-core ARM Cortex-A78AE |
| **GPU** | Orin iGPU 1024 cores @ 625 MHz (memória compartilhada CPU/GPU) |
| **DLA** | 2× NVDLA (ociosos na Fase 1) |
| **NVDEC** | 1× hardware H.264/H.265 decoder (~4-8 streams simultâneos) |
| **CSI** | 2× conectores MIPI CSI (22-pin) |
| **USB** | 4× USB 3.0 + 4× USB 2.0 |
| **Rede** | 1× Gigabit Ethernet RJ45 |
| **Software** | JetPack R36.5.0, CUDA 12.6, DeepStream 7.1 |

## Tecnologias de Captura

| Tecnologia | Conector | Formato | Decode | Latência | Distância máx |
|---|---|---|---|---|---|
| **USB UVC (C930e)** | USB 3.0 | MJPEG 1080p@30 | CPU ~49ms | ~50ms | 5m (USB ativo) |
| **USB UVC raw** | USB 3.0 | YUYV 720p@10 | Nenhum | ~1ms | 5m |
| **MIPI CSI (IMX219/477)** | Ribbon 22-pin | RAW Bayer | GPU ISP ~2ms | ~3ms | 30cm (ribbon) / 15m (coaxial) |
| **IP H.264 RTSP** | RJ45 | H.264 1080p@30 | GPU NVDEC ~3ms | ~5ms | 100m (Ethernet) |
| **IP H.265 RTSP** | RJ45 | H.265 1080p@30 | GPU NVDEC ~4ms | ~6ms | 100m |

---

## Cenário 1: MVP — 1 câmera USB

**Referencia historica do MVP de uma camera.** O runtime atual tambem suporta duas cameras USB habilitadas em paralelo.

### Configuração

```
C930e USB UVC (/dev/video0)
  → MJPEG 1920×1080 @ 15fps
  → CPU decode (49ms)
  → Pipeline: detect → quality → recognize
```

### Stack

| Componente | Detalhe |
|---|---|
| Container | 1× `iris-app` + 1× `iris-mosquitto` |
| Captura | `cv2.VideoCapture CAP_V4L2` single-thread |
| Modelos | `buffalo_m` (det_2.5g + w600k_r50 + 2d106det), TRT FP16 |
| Eventos | SQLite (embeddings) + JSONL (reconhecimentos + oclusões) |
| Capture rate | 3 capturas/s (0.333s intervalo) |
| API | FastAPI :8081 |

### Performance medida

| Métrica | Valor |
|---|---|
| Frames lidos | ~15/s |
| Capturas processadas | 3/s |
| Eventos (rosto detectado) | ~2-3/s com pessoa presente |
| CPU total | ~20% (14% decode + 6% app) |
| GPU | ~30-40% (TRT inferência) |
| RAM container | ~1.2 GB |
| Decode JPEG | 49ms (CPU, libjpeg-turbo 3.0.3) |
| Detecção | 31ms (TRT FP16, det_2.5g 480×480) |
| Pipeline total | ~97ms por captura |

### Limitações

| Limitação | Impacto |
|---|---|
| Decode JPEG em CPU | 49ms/frame, escala linear com nº de câmeras USB |
| Single-thread | 1 câmera apenas |
| Sem GStreamer nativo | Sem acesso ao pipeline acelerado NVIDIA |
| `camera_2` | Worker independente, configurado por `CAMERA_2_*` |
| USB bandwidth | 1 câmera MJPEG 1080p = ~6 MB/s — USB 3.0 sobra |

### Complexidade para próximo cenário

**Baixa** — código estável, testado em produção. Mudar para multi-câmera requer refatoração do `iris_runtime.py` (single-thread → multi-thread).

---

## Cenário 2: 2 USB + 2 IP

### Configuração

```
Câmera 1: C930e USB → MJPEG 1080p → CPU decode 49ms
Câmera 2: C930e USB → MJPEG 1080p → CPU decode 49ms
Câmera 3: IP H.264 RTSP → GPU NVDEC ~3ms
Câmera 4: IP H.264 RTSP → GPU NVDEC ~3ms
```

### Mudanças necessárias

| Arquivo | Mudança | Linhas |
|---|---|---|
| `settings.py` | Lista dinâmica de câmeras com `source_kind`, `device`, `url` | ~30 |
| `cameras.py` | Parser de lista em vez de 2 hardcoded | ~15 |
| `iris_runtime.py` | 4× threads de captura, per-camera stats/preview | ~150 |
| `main.py` | Endpoints per-camera: `/cameras/{id}/preview`, etc | ~30 |
| **Total** | | **~225 linhas** |

### Recursos

| Recurso | Consumo | Limite | Status |
|---|---|---|---|
| CPU (USB decode ×2) | ~28% | 6 cores | ✓ |
| GPU (inferência) | ~40% | 1024 cores | ✓ |
| NVDEC (IP decode ×2) | ~20% | 4-8 streams | ✓ |
| RAM | ~1.3 GB | 7.4 GB | ✓ |
| USB 3.0 | 12 MB/s | 625 MB/s | ✓ |
| Disco (eventos ×4) | ~8 JPEGs/s | NVMe 915 GB | ✓ |

### Limitações

- Cada USB adicional adiciona +14% CPU (decode MJPEG)
- 2 USB + 2 IP é o limite confortável para manter 3 capturas/s por câmera
- NVDEC saturado se IPs forem 4K

### Complexidade: **Média**

Custo de implementação: ~225 linhas, ~2 dias de desenvolvimento. Maior risco: sincronização de threads com o `_recognizer_lock`.

---

## Cenário 3: 4 USB

### Configuração

```
Câmera 1-4: C930e USB → MJPEG 1080p → CPU decode ×4
```

### Recursos

| Recurso | Consumo | Limite | Status |
|---|---|---|---|
| CPU (decode ×4) | ~56% | 6 cores | ⚠️ No limite |
| GPU (inferência) | ~40% | 1024 cores | ✓ |
| USB 3.0 | 24 MB/s | Hub 4-portas, 625 MB/s | ⚠️ Hub compartilhado |
| RAM | ~1.5 GB | 7.4 GB | ✓ |

### Limitações críticas

| Limitação | Detalhe |
|---|---|
| **CPU em 56%+** | Decode MJPEG escala linear — 4 câmeras consomem 4× 14% = 56%. Com picos (preview, encode JPEG), pode bater 70-80% e causar frame drops |
| **Hub USB compartilhado** | As 4 câmeras dividem o mesmo controlador USB 3.0. Pode haver contenção de barramento com latência adicional |
| **Throughput pipeline** | 4× 3 capturas/s = 12/s. Pipeline total por frame = 97ms. Com 4 threads, o `_recognizer_lock` serializa a inferência. Latência efetiva sobe |
| **Single point of failure** | Se o hub USB falhar, todas as câmeras caem |

### Complexidade: **Média-Alta**

Mesmo código do cenário 2 (~225 linhas), mas risco operacional alto. Não recomendado — o gargalo de CPU torna a solução frágil. Melhor migrar para MIPI/IP.

---

## Cenário 4: 1 USB + 3 IP

### Configuração

```
Câmera 1: C930e USB → MJPEG 1080p → CPU decode 49ms
Câmera 2: IP H.264 RTSP → GPU NVDEC ~3ms
Câmera 3: IP H.264 RTSP → GPU NVDEC ~3ms
Câmera 4: IP H.264 RTSP → GPU NVDEC ~3ms
```

### Recursos

| Recurso | Consumo | Limite | Status |
|---|---|---|---|
| CPU (USB decode ×1) | ~14% | 6 cores | ✓ Folgado |
| GPU (inferência) | ~40% | 1024 cores | ✓ |
| NVDEC (IP decode ×3) | ~30% | 4-8 streams | ✓ |
| RAM | ~1.3 GB | 7.4 GB | ✓ |
| Rede (3× RTSP) | ~15 Mbps | Gigabit | ✓ |
| USB 3.0 | 6 MB/s | 625 MB/s | ✓ |

### Vantagem

As 3 IPs fazem decode em GPU (NVDEC) a ~3ms cada. Só a USB paga o custo de CPU. Esse é o melhor custo-benefício para multi-câmera com o hardware atual.

### Custo de hardware adicional

| Item | Qtd | Preço unit. ~ | Total ~ |
|---|---|---|---|
| Câmera IP H.264 1080p (ex: Hikvision DS-2CD1021) | 3 | R$ 250-500 | R$ 750-1500 |
| Switch PoE 4-portas | 1 | R$ 100-200 | R$ 100-200 |
| Cabo Ethernet CAT6 | 3× 10m | R$ 30 | R$ 90 |
| **Total** | | | **R$ 940-1790** |

### Complexidade: **Média**

~225 linhas de código + configuração de IPs. Melhor cenário para deploy imediato com hardware existente.

---

## Cenário 5: 1 MIPI CSI + 3 IP

### Configuração

```
Câmera 1: MIPI CSI (IMX219/477) → RAW Bayer → GPU ISP ~2ms
Câmera 2: IP H.264 RTSP → GPU NVDEC ~3ms
Câmera 3: IP H.264 RTSP → GPU NVDEC ~3ms
Câmera 4: IP H.264 RTSP → GPU NVDEC ~3ms
```

### Recursos

| Recurso | Consumo | Limite | Status |
|---|---|---|---|
| CPU (total) | ~5% | 6 cores | ✓✓ Sobrando |
| GPU ISP (CSI) | ~5% | Dedicado | ✓ |
| GPU (inferência) | ~40% | 1024 cores | ✓ |
| NVDEC (IP ×3) | ~30% | 4-8 streams | ✓ |
| RAM | ~1.3 GB | 7.4 GB | ✓ |

### Vantagem sobre o cenário 4

Elimina o decode CPU da USB. A câmera CSI entrega RAW Bayer direto para o ISP da GPU (~2ms). Zero latência de compressão/descompressão. CPU fica livre para outras tarefas (Onix sync, MQTT, dashboard).

### Custo de hardware adicional

| Item | Qtd | Preço unit. ~ | Total ~ |
|---|---|---|---|
| IMX219 8MP MIPI CSI | 1 | R$ 80-150 | R$ 80-150 |
| Ribbon cable 22-pin 15cm | 1 | R$ 20 | R$ 20 |
| Câmera IP H.264 1080p | 3 | R$ 250-500 | R$ 750-1500 |
| Switch PoE 4-portas | 1 | R$ 100-200 | R$ 100-200 |
| Cabo Ethernet CAT6 | 3× 10m | R$ 30 | R$ 90 |
| **Total** | | | **R$ 1040-1960** |

### Limitação: distância CSI

O ribbon cable padrão tem **30cm**. Para distâncias maiores, ver Cenário 7.

### Complexidade: **Média**

~250 linhas (+ captura CSI via GStreamer `nvarguscamerasrc`). A captura CSI é nativa NVIDIA, código bem documentado.

---

## Cenário 6: 4 IP

### Configuração

```
Câmera 1-4: IP H.264 RTSP → GPU NVDEC ~3ms cada
```

### Recursos

| Recurso | Consumo | Limite | Status |
|---|---|---|---|
| CPU (total) | ~5% | 6 cores | ✓✓ |
| GPU (inferência) | ~40% | 1024 cores | ✓ |
| NVDEC (4× IP) | ~40% | 4-8 streams | ✓ |
| Rede (4× RTSP) | ~20 Mbps | Gigabit | ✓ |
| RAM | ~1.3 GB | 7.4 GB | ✓ |

### Vantagens

- Pipeline 100% GPU — CPU quase ociosa
- Distância até 100m por câmera (Ethernet)
- PoE: 1 cabo para energia + dados
- Sem USB, sem ribbon, sem interferência de barramento
- Câmeras IP são hardware padrão de mercado (Hikvision, Dahua, Intelbras)

### Custo de hardware adicional

| Item | Qtd | Preço unit. ~ | Total ~ |
|---|---|---|---|
| Câmera IP H.264 1080p | 4 | R$ 250-500 | R$ 1000-2000 |
| Switch PoE 8-portas | 1 | R$ 150-300 | R$ 150-300 |
| Cabo Ethernet CAT6 | 4× 10m | R$ 30 | R$ 120 |
| **Total** | | | **R$ 1270-2420** |

### Complexidade: **Média-Baixa**

~200 linhas (sem código de captura USB). Menos branches de código porque todas as câmeras usam o mesmo `source_kind=gst_rtsp`. Código mais limpo.

---

## Cenário 7: 4 MIPI CSI

### Configuração

```
Câmera 1-2: MIPI CSI conectores onboard (2× disponíveis)
Câmera 3-4: MIPI CSI via adaptador adicional
```

### Limitação física crítica

O Orin Nano Dev Kit tem **apenas 2 conectores CSI onboard**. Para 4 câmeras CSI, é necessário hardware adicional.

### Opções de expansão CSI

| Opção | Descrição | Custo ~ |
|---|---|---|
| **A. CSI MIPI switch/mux board** | Placa que multiplexa 4→2 CSI. Suportado pelo JetPack com `nvarguscamerasrc`. Latência adicional ~1ms | R$ 300-600 |
| **B. USB-to-CSI bridge** | Converte CSI para USB 3.0 — perde a vantagem do CSI | R$ 200-400 |
| **C. Jetson Orin NX/Nano com mais CSI** | Orin NX tem 4-6 CSI dependendo do carrier board | R$ 2000-4000 |

### Distância: o problema real

O ribbon cable CSI padrão tem **30cm**. Para distâncias de 5-15m:

| Extensor | Descrição | Distância máx | Custo ~ |
|---|---|---|---|
| **FPC coaxial extender** | Cabo coaxial blindado com terminador CSI. Mantém integridade do sinal | 10m | R$ 80-150/cabo |
| **GMSL serializer/deserializer** | Converte CSI → coaxial GMSL → CSI. Usado em automotive-grade. NVIDIA suporta via driver MAX9295/MAX9296 | 15m | R$ 300-500/par |
| **CSI-to-RJ45 bridge** | Microcontrolador converte CSI → UDP sobre Ethernet → reconstitui no Jetson. Não é GStreamer nativo, requer software customizado | 100m | R$ 400-800 |
| **CSI-to-HDMI adapter + HDMI extender** | CSI → HDMI → extensor HDMI sobre Ethernet → HDMI → framegrabber. **Gambiarra.** Latência alta (~50-100ms), custo ~R$ 300-600 | 50m | R$ 300-600 |

### Recomendação para >30cm

**Para câmeras a mais de 30cm do Jetson, use câmeras IP.** MIPI CSI foi projetado para curta distância (dentro do gabinete). Estender CSI a 10m+ custa mais que a câmera IP equivalente e tem latência e complexidade piores.

### Custo total 4 CSI com extensão a 10m

| Item | Qtd | Preço unit. ~ | Total ~ |
|---|---|---|---|
| IMX219 8MP MIPI CSI | 4 | R$ 80-150 | R$ 320-600 |
| CSI switch/mux board 4→2 | 1 | R$ 300-600 | R$ 300-600 |
| FPC coaxial extender 10m | 4 | R$ 80-150 | R$ 320-600 |
| Ribbon cable 15cm | 4 | R$ 20 | R$ 80 |
| **Total** | | | **R$ 1020-1880** |

### Complexidade: **Alta**

GStreamer nativo para múltiplas CSI com `nvarguscamerasrc` requer configuração por device tree e pode precisar de kernel recompile. Adicionar extensores coaxial GMSL traz mais drivers. **Só justificável se latência < 5ms for requisito absoluto** (ex: robótica, controle em tempo real).

---

## Resumo Comparativo

| Cenário | Câmeras | Decode CPU | Decode GPU | CPU livre | Complexidade código | Custo hardware | Recomendado? |
|---|---|---|---|---|---|---|---|
| 1. 1 USB | 1× USB | 49ms | - | 80% | **Baixa** (feito) | R$ 0 | ✅ MVP atual |
| 2. 2 USB + 2 IP | 2× USB, 2× IP | 98ms | 6ms | 44% | Média (225 ln) | R$ 700-1500 | ⚠️ OK |
| 3. 4 USB | 4× USB | 196ms | - | 30% | Média (225 ln) | R$ 0-400 | ❌ CPU frágil |
| 4. 1 USB + 3 IP | 1× USB, 3× IP | 49ms | 9ms | 66% | Média (225 ln) | R$ 940-1790 | ✅ Melhor custo-benefício |
| 5. 1 CSI + 3 IP | 1× CSI, 3× IP | - | 11ms | 95% | Média (250 ln) | R$ 1040-1960 | ✅ Ótimo, zero CPU |
| 6. 4 IP | 4× IP | - | 12ms | 95% | Média-Baixa (200 ln) | R$ 1270-2420 | ✅✅ Melhor arquitetura |
| 7. 4 CSI 10m | 4× CSI | - | ~8ms | 95% | Alta (+ drivers) | R$ 1020-1880 | ❌ Só se latência < 5ms for requisito |

## Recomendação

1. **Hoje**: Manter cenário 1 (MVP). Já está em produção, estável.
2. **Próximo passo**: Cenário 6 (4 IP). Remove 100% do decode CPU, usa hardware NVIDIA nativo, distância de 100m com PoE (1 cabo), código mais limpo, câmeras padrão de mercado.
3. **Transição**: Se quiser manter a C930e existente, cenário 4 (1 USB + 3 IP). Migra gradualmente.
4. **Futuro**: MIPI CSI só se latência < 5ms for requisito (não é o caso de reconhecimento facial em fila).
