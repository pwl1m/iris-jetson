from .schemas import CameraConfig
from .settings import Settings


def configured_cameras(settings: Settings) -> list[CameraConfig]:
    # Cameras 1-2 possuem workers independentes quando habilitadas. As demais
    # continuam rastreaveis para preservar o contrato de ate quatro cameras.
    specs = [
        (
            settings.camera_1_id,
            settings.camera_1_stream_url or settings.stream_url,
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
    for camera_id, stream_url, enabled, primary, source_kind, device, serial, model, stable_path in specs:
        effective_source_kind = source_kind or ("rtsp" if stream_url else None)
        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                stream_url=stream_url,
                enabled=bool(enabled and (primary or stream_url or device)),
                primary=primary,
                source_kind=effective_source_kind,
                device=device or None,
                input_format=settings.usb_camera_input_format
                if effective_source_kind in {"jetson_gst_usb", "usb"}
                else None,
                width=settings.usb_camera_width if effective_source_kind in {"jetson_gst_usb", "usb"} else None,
                height=settings.usb_camera_height if effective_source_kind in {"jetson_gst_usb", "usb"} else None,
                fps=settings.usb_camera_fps if effective_source_kind in {"jetson_gst_usb", "usb"} else None,
                serial_number=serial or None,
                model_name=model or None,
                stable_path=stable_path or None,
            )
        )
    return [camera for camera in cameras if camera.camera_id]
