from .schemas import CameraConfig
from .settings import Settings


def configured_cameras(settings: Settings) -> list[CameraConfig]:
    cameras = [
        CameraConfig(
            camera_id=settings.camera_1_id,
            stream_url=settings.camera_1_stream_url or settings.stream_url,
            enabled=bool(settings.camera_1_enabled),
            primary=True,
        ),
        CameraConfig(
            camera_id=settings.camera_2_id,
            stream_url=settings.camera_2_stream_url,
            enabled=bool(settings.camera_2_enabled and settings.camera_2_stream_url),
            primary=False,
        ),
    ]
    return [camera for camera in cameras if camera.camera_id]
