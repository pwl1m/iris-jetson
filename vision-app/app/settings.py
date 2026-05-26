from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_db_path: str = "/data/faces/faces.sqlite3"
    face_model_name: str = "buffalo_l"
    face_model_root: str = "/models/face"
    face_det_size: str = "640,640"
    face_ctx_id: int = 0
    face_similarity_threshold: float = 0.55
    face_max_results: int = 5
    face_providers: str = "CPUExecutionProvider"

    worker_log_level: str = "INFO"
    event_log_path: str = "/data/events/recognitions.jsonl"
    capture_dir: str = "/data/events/captures"
    camera_name: str = "entrada"
    stream_url: str = "rtsp://go2rtc:8554/usb_camera"
    stream_worker_enabled: bool = True
    stream_capture_interval_seconds: float = 1.0
    stream_reconnect_delay_seconds: float = 3.0
    stream_jpeg_quality: int = 90
    stream_min_face_score: float = 0.5
    stream_max_capture_files: int = 2000
    stream_preview_update_interval_seconds: float = 0.25
    stream_preview_max_width: int = 640
    stream_preview_jpeg_quality: int = 70
    stream_reader_buffer_size: int = 1

    @property
    def det_size_tuple(self) -> tuple[int, int]:
        width, height = self.face_det_size.split(",", 1)
        return int(width), int(height)

    @property
    def providers_list(self) -> list[str]:
        return [item.strip() for item in self.face_providers.split(",") if item.strip()]


settings = Settings()
