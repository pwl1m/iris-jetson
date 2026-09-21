#!/usr/bin/env python3
"""Linha de base do cenário: mede o que a iluminação e o enquadramento mudam.

Por que existe
--------------
Os limiares de qualidade e de oclusão do Iris foram calibrados na linha USB, com
outra câmera, outro enquadramento e outra iluminação. Vários deles são absolutos
e portanto dependentes de cena:

    VISUAL_OCCLUSION_DARK_PIXEL_THRESHOLD   (luminância absoluta)
    VISUAL_OCCLUSION_SKIN_CR_MIN/MAX        (faixa YCrCb de pele)
    VISUAL_OCCLUSION_SKIN_CB_MIN/MAX
    FACE_MIN_BLUR_SCORE                     (variância de Laplaciano)
    VISUAL_OCCLUSION_VERDICT_MIN_SIZE       (tamanho de rosto, depende da posição)

Trocar de cenário não invalida o pipeline, mas invalida esses números. Este
script lê os eventos que o deploy já gravou e devolve a distribuição real do
cenário novo, ao lado da referência da linha USB, para que a recalibração seja
uma leitura e não uma pesquisa.

Não altera nada. Só lê a API.

Uso
---
    python3 scripts/scene_baseline.py --url http://127.0.0.1:8181 --limit 2000

Rode depois de o cenário novo ter acumulado eventos suficientes: com menos de
~200 rostos as caudas não significam nada.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


# Referência da linha USB, medida em 19/09/2026 sobre 400 frames retidos
# amostrados ao acaso, com o pipeline real: representa tráfego típico daquele
# cenário. Serve para enxergar o DESLOCAMENTO ao trocar de cena, nunca como alvo.
#
# Cuidado ao comparar com outra fonte: os 2.542 eventos históricos que carregam
# métricas completas passaram pelo piso antigo de 96 px, então são só rostos
# grandes (largura p50 de 119 px, skin_ratio p50 de 0,391) e não descrevem o
# tráfego típico. Ver docs/ip-camera/13_OCCLUSION_EVIDENCE_AND_RESOLUTION.md.
REFERENCIA_USB = {
    "largura_bbox": ("mediana", 51.0),
    "crop_luminance": ("mediana normal", 88.4),
    "skin_ratio": ("mediana normal", 0.437),
    "blur": ("mediana normal", 740.1),
    "det_score": ("mediana", 0.742),
    "dark_lower_ratio": ("mediana normal", 0.311),
}


def buscar(url: str, limit: int) -> list[dict]:
    req = urllib.request.Request(f"{url.rstrip('/')}/events?limit={limit}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp).get("events", [])
    except urllib.error.URLError as exc:
        sys.exit(f"nao foi possivel ler {url}: {exc}")


def extrair(eventos: list[dict]) -> list[dict]:
    linhas = []
    for evento in eventos:
        face = ((evento.get("recognition") or {}).get("face")) or {}
        visual = face.get("visual_occlusion") or {}
        metricas = visual.get("metrics") or {}
        bbox = face.get("bbox") or []
        if len(bbox) != 4:
            continue
        linhas.append({
            "largura_bbox": float(bbox[2]) - float(bbox[0]),
            "det_score": face.get("det_score"),
            "crop_luminance": metricas.get("crop_luminance"),
            "skin_ratio": metricas.get("skin_ratio"),
            "dark_lower_ratio": metricas.get("dark_lower_ratio"),
            "dark_asymmetry": metricas.get("dark_asymmetry"),
            "blur": metricas.get("blur") or (evento.get("quality") or {}).get("blur"),
            "yaw": metricas.get("yaw"),
            "eye_nose_asymmetry": metricas.get("eye_nose_asymmetry"),
            "suspeito": bool(visual.get("suspected")),
            "abaixo_do_piso": "below_verdict_floor" in (visual.get("signals") or []),
        })
    return linhas


def percentil(valores: list[float], q: float) -> float:
    if not valores:
        return float("nan")
    ordenados = sorted(valores)
    posicao = (len(ordenados) - 1) * q / 100.0
    baixo = int(posicao)
    alto = min(baixo + 1, len(ordenados) - 1)
    return ordenados[baixo] + (ordenados[alto] - ordenados[baixo]) * (posicao - baixo)


def coluna(linhas: list[dict], chave: str) -> list[float]:
    return [r[chave] for r in linhas if isinstance(r.get(chave), (int, float))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8181")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--piso", type=int, default=64,
                        help="VISUAL_OCCLUSION_VERDICT_MIN_SIZE em uso no deploy "
                             "(padrao 64, era 96 antes de 21/09/2026)")
    args = parser.parse_args()

    linhas = extrair(buscar(args.url, args.limit))
    if not linhas:
        sys.exit("nenhum evento com metricas. O cenario ja rodou com pessoas?")

    print(f"eventos com metricas: {len(linhas)}")
    if len(linhas) < 200:
        print("AVISO: menos de 200 rostos. As caudas nao sao confiaveis ainda.\n")

    print(f"\n{'metrica':<22}{'p05':>9}{'p50':>9}{'p95':>9}{'ref USB':>11}{'desvio':>9}")
    for chave, (_, referencia) in REFERENCIA_USB.items():
        valores = coluna(linhas, chave)
        if not valores:
            print(f"{chave:<22}{'sem dado':>38}")
            continue
        mediana = percentil(valores, 50)
        razao = mediana / referencia if referencia else float("nan")
        print(f"{chave:<22}{percentil(valores,5):>9.3f}{mediana:>9.3f}"
              f"{percentil(valores,95):>9.3f}{referencia:>11.3f}{razao:>8.2f}x")

    larguras = coluna(linhas, "largura_bbox")
    # O piso e configuravel e ja mudou uma vez (96 -> 64 em 21/09/2026). Fixa-lo
    # aqui fazia o relatorio contradizer o deploy: com 64 rodando, a linha dizia
    # 20,2% de cobertura quando a real era 43,8%.
    piso = args.piso
    faixas = sorted({0, 48, 64, 96, 160, piso})
    for i, baixo in enumerate(faixas):
        alto = faixas[i + 1] if i + 1 < len(faixas) else 10**6
        n = sum(1 for w in larguras if baixo <= w < alto)
        rotulo = f"{baixo}-{alto} px" if alto < 10**6 else f">= {baixo} px"
        marca = "  <- piso" if baixo == piso else ""
        print(f"  {rotulo:<14}{n:>7}  {n/len(larguras)*100:>5.1f}%{marca}")
    acima = sum(1 for w in larguras if w >= piso)
    print(f"  -> {acima/len(larguras)*100:.1f}% dos rostos podem receber veredito."
          f"  Na linha USB eram 0,6%.")
    mediana_largura = percentil(larguras, 50)
    if mediana_largura < piso:
        print(f"  ATENCAO: a largura MEDIANA e {mediana_largura:.0f} px, abaixo do")
        print(f"  piso de {piso}. Mais da metade do trafego nao recebe veredito, e")
        print(f"  baixar mais o piso nao resolve: {piso} px ja e pouca evidencia.")
        print(f"  Esta largura e em pixels da FONTE, nao do detector -- entao")
        print(f"  mexer em FACE_DET_SIZE ou recortar a entrada do detector nao a")
        print(f"  muda. So otica (aproximar/zoom) ou um stream de resolucao maior.")
        print(f"  Ver docs/ip-camera/14_CADASTRO_E_LINHA_DE_BASE.md.")

    suspeitos = sum(1 for r in linhas if r["suspeito"])
    sob_piso = sum(1 for r in linhas if r["abaixo_do_piso"])
    print(f"\noclusao: {suspeitos} suspeitos ({suspeitos/len(linhas)*100:.2f}%),"
          f" {sob_piso} medidos mas abaixo do piso de veredito")

    print("\ncomo usar este relatorio")
    print("  crop_luminance muito abaixo de 88 -> a cena e mais escura que a USB e")
    print("    VISUAL_OCCLUSION_DARK_PIXEL_THRESHOLD=55 passa a marcar cena normal.")
    print("  skin_ratio muito abaixo de 0,437 -> as faixas YCrCb nao valem sob esta")
    print("    luz; recalibrar antes de confiar no sinal low_skin_visibility.")
    print("  blur muito abaixo de 740 -> FACE_MIN_BLUR_SCORE=40 fica perto do normal")
    print("    da cena e passa a descartar rosto bom.")
    print("  se a faixa acima do piso crescer bem acima de 0,6%, o piso ja cobre")
    print("    trafego real e VISUAL_OCCLUSION_VERDICT_MIN_SIZE pode cair de novo.")


if __name__ == "__main__":
    main()
