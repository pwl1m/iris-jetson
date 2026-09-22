"""Regiao de interesse (ROI) por camera: filtra o censo de pessoas por zona.

A ROI e opcional e expressa em FRACOES normalizadas (0.0 a 1.0), nunca em
pixel absoluto. Isso e deliberado: a resolucao da camera ja mudou uma vez
nesta instalacao (1920x1080 -> 2560x1440, ver D-019 em docs/DECISIONS.md) e
uma ROI em pixel absoluto teria virado lixo silenciosamente nessa troca. Em
fracao, a mesma zona ("o vao da porta", "a metade direita do quadro")
continua correta em qualquer resolucao que a camera entregar.

A ROI filtra apenas o CENSO de pessoas (`frame_census` / `/people-count`).
Ela nao toca no pipeline de reconhecimento, tracking ou oclusao, que
continuam vendo o frame inteiro exatamente como hoje -- ligar uma ROI nao
muda nenhum comportamento ja calibrado em producao, so decide o que entra
no contador novo.
"""

from __future__ import annotations

Roi = tuple[float, float, float, float]


def parse_roi(value: str | None) -> Roi | None:
    """Converte "x1,y1,x2,y2" (fracoes de 0 a 1) numa tupla validada.

    Retorna None para vazio ou configuracao invalida. Nunca levanta, no
    mesmo espirito de `cameras.stream_render_mode`: uma ROI mal configurada
    nao pode derrubar o censo, so desliga o filtro e deixa o frame inteiro
    contar, que e o comportamento seguro por omissao.
    """
    text = (value or "").strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(part.strip()) for part in parts)
    except ValueError:
        return None
    if not all(0.0 <= coord <= 1.0 for coord in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_center_inside(bbox: list[float], roi: Roi, frame_width: int, frame_height: int) -> bool:
    """True se o centro do bbox (em pixels) cai dentro da ROI.

    Usa o centro do bbox, nao a intersecao de area, de proposito: uma pessoa
    cujo rosto esta so parcialmente dentro da zona (por exemplo cruzando a
    borda do vao da porta) ainda deve contar como "dentro" ou "fora" de forma
    binaria e previsivel, sem um limiar de IoU adicional para calibrar.
    """
    if frame_width <= 0 or frame_height <= 0:
        return True
    x1, y1, x2, y2 = roi
    left, top = x1 * frame_width, y1 * frame_height
    right, bottom = x2 * frame_width, y2 * frame_height
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return left <= center_x <= right and top <= center_y <= bottom
