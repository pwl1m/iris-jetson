# People Count API — contrato para quem for integrar

Backend pronto do lado do Jetson. **Ainda não foi validado com câmera real
nem implantado** — foi implementado e testado só com testes unitários (sem
hardware), porque o pedido era "prepare o back-end, vou trabalhar com isso
depois". Antes de depender disso em produção, rode pelo menos uma passagem
real de pessoas e confira `GET /people-count` e `GET /crowd` batendo.

Contexto de por que isso existe e como foi decidido: `docs/DECISIONS.md`
(D-020) e a conversa que originou o pedido — contagem de rostos numa foto
específica marcada como oclusão, e a possibilidade de restringir a contagem
a uma zona do quadro.

## O que é, o que não é

- Conta **rostos detectados** (reaproveita o SCRFD que já roda no pipeline),
  não corpos/pessoas de costas. Mesma limitação que `/crowd` já documenta.
- **Não** reconhece nem cadastra ninguém — não toca em `face_embeddings`,
  não grava evento em `recognitions.jsonl`/`occlusions.jsonl`.
- É **persistido** (tabela SQLite `frame_census`), ao contrário de `/crowd`,
  que é só memória e zera a cada restart do worker.
- Não deduplica tracks fragmentados: se o tracker perder e recriar o track
  da mesma pessoa (acontece com capacete, cabeça muito inclinada, etc. — ver
  exemplos reais analisados na sessão que originou isso), cada aparição do
  detector conta separadamente no frame em que ocorreu. `face_count` é
  "quantos rostos o detector viu **neste frame**", não "quantas pessoas
  distintas passaram numa janela de tempo".

## Endpoints

### `POST /people-count`

Conta rostos numa foto avulsa e grava uma linha no histórico.

```
POST /people-count?camera_id=entrada
Content-Type: multipart/form-data
file: <jpeg ou png>
```

`camera_id` é opcional:
- Se **omitido**: conta o frame inteiro, sem ROI.
- Se **informado** e a câmera tiver `CAMERA_n_ROI` configurada: `face_count`
  reflete só a zona, e cada item de `boxes` ganha `inside_roi` (`true`/`false`).
- Se informado mas a câmera não existir no inventário: tratado como "sem
  ROI" (não dá erro — `camera_id` também é gravado como veio, mesmo que não
  corresponda a nenhuma câmera configurada; isso é intencional para permitir
  contar fotos de fontes externas sob um rótulo arbitrário).

Resposta:

```json
{
  "census_id": 42,
  "camera_id": "entrada",
  "captured_at": "2026-09-22T14:10:00.000000+00:00",
  "face_count": 3,
  "faces_detected_total": 4,
  "roi_applied": true,
  "boxes": [
    {"bbox": [119.0, 348.5, 245.2, 493.8], "det_score": 0.87, "inside_roi": true},
    {"bbox": [900.0, 120.0, 1010.0, 260.0], "det_score": 0.79, "inside_roi": false}
  ]
}
```

`faces_detected_total` é sempre o total bruto detectado no frame inteiro,
mesmo com ROI ligada — serve para saber quanto a zona está descartando.
`face_count` é o que a ROI deixou passar (== `faces_detected_total` sem ROI).

Erros: `422` se o arquivo não é uma imagem decodificável.

### `GET /people-count`

Histórico persistido.

```
GET /people-count?limit=20&camera_id=entrada&since=37
```

- `limit`: default 20, teto 500.
- `camera_id`: filtra por câmera; omitido lista todas.
- `since`: cursor incremental — o `id` da última linha já vista. **Sem**
  `since`, devolve as `limit` linhas mais recentes (mais nova primeiro).
  **Com** `since`, devolve só linhas com `id` maior, em ordem cronológica
  (mesmo padrão de paginação incremental que `GET /events` já usa).

```json
{
  "census": [
    {
      "id": 43,
      "camera_id": "entrada",
      "captured_at": "2026-09-22T14:10:05.123456+00:00",
      "capture_number": 570036,
      "face_count": 2,
      "roi_applied": 0,
      "source": "worker",
      "created_at": "2026-09-22 14:10:05"
    }
  ]
}
```

`source` é `"worker"` (uma linha por frame processado pelo worker de câmera
ao vivo, gravada automaticamente — sem chamada nenhuma da sua parte) ou
`"upload"` (veio de um `POST /people-count`). `capture_number` só existe em
linhas `"worker"`; em `"upload"` vem `null`.

**Atenção de volume**: com o worker ligado, o padrão hoje grava uma linha
por frame processado. Na cadência atual da linha IP (5 FPS, ver D-003 em
`docs/DECISIONS.md`) isso é ~5 linhas/segundo = ~432.000 linhas/dia por
câmera. A tabela não tem rotina de purga automática (a retenção de captures/
eventos em `STREAM_CAPTURE_RETENTION_HOURS` não cobre esta tabela). Se for
manter isto ligado por muito tempo, quem for consumir precisa decidir uma
política de retenção antes — não veio pronta.

## Configurar a ROI (zona que filtra a contagem)

Variável de ambiente por câmera, formato `"x1,y1,x2,y2"`, **frações de 0 a
1**, nunca pixel absoluto:

```
CAMERA_1_ROI=0.30,0.10,0.90,0.95
CAMERA_2_ROI=
CAMERA_3_ROI=
CAMERA_4_ROI=
```

- Vazio (default) = sem filtro, conta o frame inteiro.
- `x1,y1` é o canto superior-esquerdo da zona, `x2,y2` o inferior-direito,
  ambos como fração da largura/altura do frame.
- Valor inválido (não numérico, fora de 0–1, ou `x2<=x1`/`y2<=y1`) **nunca
  derruba o serviço** — cai em "sem filtro" silenciosamente. Não há log de
  erro para isso hoje; se for depurar uma ROI que parece não estar
  aplicando, confira o valor bruto da env var primeiro.
- **Por que fração e não pixel**: a resolução desta câmera já mudou uma vez
  nesta instalação (1920×1080 → 2560×1440, D-019 em `docs/DECISIONS.md`,
  aplicado 22/09/2026). Uma ROI em pixel teria ficado errada silenciosamente
  nessa troca. Em fração, a mesma zona ("o vão da porta", por exemplo)
  continua correta em qualquer resolução.
- A ROI usa o **centro do bbox** do rosto, não interseção de área: um rosto
  cruzando a borda da zona conta de forma binária (dentro ou fora), sem
  limiar de sobreposição para calibrar.
- **A ROI filtra só este contador.** Reconhecimento, tracking e detecção de
  oclusão continuam recebendo o frame inteiro e não mudam de comportamento
  ao ligar uma ROI — é uma decisão deliberada (ver D-020) para não alterar
  nada que já está calibrado em produção.

## Onde os dados ficam

Tabela nova `frame_census`, em arquivo SQLite próprio — **não** dentro de
`faces.sqlite3` (que é só biometria):

```
FRAME_CENSUS_DB_PATH=/data/events/census.sqlite3   # default, raramente precisa mudar
```

Schema:

```sql
CREATE TABLE frame_census (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    capture_number INTEGER,      -- NULL quando source='upload'
    face_count INTEGER NOT NULL,
    roi_applied INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'worker',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Criada automaticamente no boot (`CREATE TABLE IF NOT EXISTS`, mesmo padrão
de `face_embeddings` em `app/storage.py`) — **não precisa de migração
manual**, só precisa que o volume `/data/events` já exista (já existe, é o
mesmo das capturas/eventos atuais).

## Para subir isto

1. Nenhuma mudança de `docker-compose*.yml` é obrigatória — os defaults
   (sem ROI) mantêm o comportamento atual e a tabela se cria sozinha.
2. Se quiser ROI: adicionar `CAMERA_n_ROI` ao `.env` (ou ao compose) antes
   de subir o container — variável lida uma vez, no boot.
3. **Precisa rebuild da imagem** (`iris-app:ip-dev` ou equivalente): o
   código novo (`app/roi.py`, mudanças em `app/settings.py`, `app/schemas.py`,
   `app/cameras.py`, `app/storage.py`, `app/iris_runtime.py`, `app/main.py`)
   está no diretório do build context, não em volume montado.
4. Depois de subir, validar:
   - `POST /people-count` com uma foto qualquer devolve `face_count` coerente.
   - `GET /people-count` mostra a linha gravada.
   - Deixar o worker rodar um pouco e conferir que `frame_census` está
     recebendo linhas `source="worker"` (uma por frame, ver aviso de volume
     acima antes de deixar ligado por muito tempo).
   - Se configurou ROI, comparar `face_count` (dentro da zona) contra
     `faces_detected_total` (frame inteiro) numa foto com gente fora da
     zona de propósito.

## O que falta / não veio pronto

- **Sem validação em câmera real.** Todo o trabalho foi feito com os
  containers do Jetson parados (pedido explícito da sessão); só há testes
  unitários (`tests/ip_camera/test_people_count.py`, 16 casos, sem
  dependência de hardware/ML).
- **Sem rotina de purga** para `frame_census` — ver aviso de volume acima.
- **Sem deduplicação de pessoa por janela de tempo** — só conta por frame.
  Se o objetivo real for "quantas pessoas distintas passaram nos últimos N
  segundos" (em vez de "quantos rostos neste frame"), isso é uma segunda
  camada de lógica sobre esta tabela (agrupar por proximidade temporal +
  similaridade de embedding), não construída aqui.
- **ROI não testada com coordenadas de uma cena real** — os valores de
  exemplo acima são ilustrativos. A câmera mudou de posição recentemente
  (ver correção de 22/09/2026 em D-019, `docs/DECISIONS.md`) e ainda não
  tem baseline de cena atual; definir a ROI real só faz sentido depois do
  `scripts/scene_baseline.py` rodar no enquadramento definitivo.
- **Sem entrada nos dashboards HTML nem no `_DOCS_HTML` de `app/main.py`**
  — só API crua por enquanto.
