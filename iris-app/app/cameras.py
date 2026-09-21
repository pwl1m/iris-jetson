from .schemas import CameraConfig
from .settings import Settings


def public_stream_urls(lan_url: str, tailscale_url: str, legacy_url: str) -> dict[str, str]:
    """Return the browser-facing transports without publishing empty values.

    ``CAMERA_n_PUBLIC_STREAM_URL`` remains the LAN fallback during migration so
    existing Onix installations continue to receive the exact same stream URL.
    """
    urls: dict[str, str] = {}
    lan_url = lan_url.strip()
    tailscale_url = tailscale_url.strip()
    legacy_url = legacy_url.strip()

    if lan_url or legacy_url:
        urls["lan"] = lan_url or legacy_url
    if tailscale_url:
        urls["tailnet"] = tailscale_url
    return urls


# O ViewCare resolve a URL publica e escolhe o player pelo modo de renderizacao.
# Hoje `IrisController::getCameraStreamAction` devolve 'mjpeg' fixo, porque era o
# unico transporte que existia. Publicar a dica aqui e aditivo: o Onix ignora
# campo desconhecido, e passa a usar quando o patch do lado dele for aplicado.
RENDER_MODES = {
    "mjpeg": "multipart/x-mixed-replace",
    "hls": "application/vnd.apple.mpegurl",
    "fmp4": "video/mp4",
}


def stream_render_mode(mode: str | None) -> tuple[str, str]:
    """Modo de renderizacao e content-type da URL publicada.

    Desconhecido ou vazio cai em `mjpeg`, que e o que o ViewCare ja sabe tocar.
    Nunca levanta: uma configuracao errada nao pode derrubar o inventario.
    """
    chave = (mode or "").strip().lower()
    if chave not in RENDER_MODES:
        chave = "mjpeg"
    return chave, RENDER_MODES[chave]


def inventory_stream_url(source_kind: str | None, public_stream_url: str | None, source_stream_url: str) -> str:
    """Return only a browser-facing URL for the external camera inventory.

    The IP engine endpoint is an authenticated, container-private frame
    contract.  It must never be handed to Onix/ViewCare as if it were a
    browser preview URL.
    """
    public_url = (public_stream_url or "").strip()
    if public_url:
        return public_url
    if (source_kind or "").strip().lower() == "ip_engine":
        return ""
    return (source_stream_url or "").strip()


def configured_cameras(settings: Settings) -> list[CameraConfig]:
    # Cameras 1-2 possuem workers independentes quando habilitadas. As demais
    # continuam rastreaveis para preservar o contrato de ate quatro cameras.
    specs = [
        (
            settings.camera_1_id,
            settings.camera_1_stream_url or settings.stream_url,
            settings.camera_1_public_stream_url,
            settings.camera_1_lan_stream_url,
            settings.camera_1_tailscale_stream_url,
            settings.camera_1_enabled,
            True,
            settings.camera_1_source_kind or settings.stream_source_kind_normalized,
            settings.camera_1_device,
            settings.camera_1_serial,
            settings.camera_1_model,
            settings.camera_1_stable_path,
        ),
        (
            settings.camera_2_id,
            settings.camera_2_stream_url,
            settings.camera_2_public_stream_url,
            settings.camera_2_lan_stream_url,
            settings.camera_2_tailscale_stream_url,
            settings.camera_2_enabled,
            False,
            settings.camera_2_source_kind,
            settings.camera_2_device,
            settings.camera_2_serial,
            settings.camera_2_model,
            settings.camera_2_stable_path,
        ),
        (
            settings.camera_3_id,
            settings.camera_3_stream_url,
            settings.camera_3_public_stream_url,
            settings.camera_3_lan_stream_url,
            settings.camera_3_tailscale_stream_url,
            settings.camera_3_enabled,
            False,
            settings.camera_3_source_kind,
            settings.camera_3_device,
            "",
            "",
            "",
        ),
        (
            settings.camera_4_id,
            settings.camera_4_stream_url,
            settings.camera_4_public_stream_url,
            settings.camera_4_lan_stream_url,
            settings.camera_4_tailscale_stream_url,
            settings.camera_4_enabled,
            False,
            settings.camera_4_source_kind,
            settings.camera_4_device,
            "",
            "",
            "",
        ),
    ]

    cameras = []
    for camera_id, stream_url, public_stream_url, lan_stream_url, tailscale_stream_url, enabled, primary, source_kind, device, serial, model, stable_path in specs:
        effective_source_kind = source_kind or ("rtsp" if stream_url else None)
        usb_source = effective_source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "http_mjpeg"}
        effective_device = (device or settings.usb_camera_device) if usb_source else (device or None)
        stream_urls = public_stream_urls(lan_stream_url, tailscale_stream_url, public_stream_url)
        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                stream_url=stream_url,
                enabled=bool(enabled and (primary or stream_url or device)),
                public_stream_url=stream_urls.get("lan") or public_stream_url or None,
                public_stream_urls=stream_urls,
                primary=primary,
                source_kind=effective_source_kind,
                device=effective_device,
                input_format=settings.usb_camera_input_format
                if effective_source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "http_mjpeg"}
                else None,
                width=settings.usb_camera_width if effective_source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "http_mjpeg"} else None,
                height=settings.usb_camera_height if effective_source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "http_mjpeg"} else None,
                fps=settings.usb_camera_fps if effective_source_kind in {"jetson_gst_usb", "usb", "gst_usb_sampled", "http_mjpeg"} else None,
                serial_number=serial or None,
                model_name=model or None,
                stable_path=stable_path or None,
            )
        )
    return [camera for camera in cameras if camera.camera_id]
