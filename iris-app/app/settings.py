from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_db_path: str = "/data/faces/faces.sqlite3"
    face_model_name: str = "buffalo_m"
    face_model_root: str = "/models/face"
    face_det_size: str = "480,480"
    face_ctx_id: int = 0
    face_similarity_threshold: float = 0.55
    face_max_results: int = 5
    face_providers: str = "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
    face_trt_fp16: bool = True
    face_trt_engine_cache_path: str = "/data/trt-engines"

    worker_log_level: str = "INFO"
    event_log_path: str = "/data/events/recognitions.jsonl"
    occlusion_log_path: str = "/data/events/occlusions.jsonl"
    capture_dir: str = "/data/events/captures"
    face_crop_dir: str = "/data/events/faces"
    camera_name: str = "entrada"
    stream_source_kind: str = "rtsp"
    stream_url: str = "rtsp://iris-go2rtc:8554/usb_camera"
    stream_gst_pipeline: str = ""
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
    stream_source_ready_timeout_seconds: float = 20.0
    stream_source_probe_url: str = "http://iris-go2rtc:1984/api/streams"
    pipeline_checks_per_second: float = 1.0
    pipeline_single_face: bool = True
    pipeline_save_face_crop: bool = True
    pipeline_face_crop_padding: float = 0.25
    face_min_det_score: float = 0.65
    face_min_width: int = 48
    face_min_height: int = 48
    face_min_blur_score: float = 40.0
    camera_1_id: str = "entrada"
    camera_1_stream_url: str = "rtsp://iris-go2rtc:8554/usb_camera"
    camera_1_enabled: bool = True
    camera_2_id: str = "entrada_2"
    camera_2_stream_url: str = ""
    camera_2_enabled: bool = False
    usb_camera_device: str = "/dev/video0"
    usb_camera_input_format: str = "mjpeg"
    usb_camera_width: int = 1920
    usb_camera_height: int = 1080
    usb_camera_fps: int = 15

    @property
    def det_size_tuple(self) -> tuple[int, int]:
        width, height = self.face_det_size.split(",", 1)
        return int(width), int(height)

    @property
    def providers_list(self) -> list[str]:
        return [item.strip() for item in self.face_providers.split(",") if item.strip()]

    @property
    def stream_source_kind_normalized(self) -> str:
        return self.stream_source_kind.strip().lower()


settings = Settings()
