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
            settings.camera_1_device or settings.usb_camera_device,
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
                device=device or None,
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
