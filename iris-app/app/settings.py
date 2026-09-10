from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_db_path: str = "/data/faces/faces.sqlite3"
    reference_image_dir: str = "/data/faces/reference-images"
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
    stream_capture_retention_hours: float = 48.0
    stream_preview_update_interval_seconds: float = 0.25
    stream_preview_enabled: bool = True
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
    visual_occlusion_enabled: bool = True
    visual_occlusion_score_threshold: float = 0.75
    visual_occlusion_min_width: int = 96
    visual_occlusion_min_height: int = 96
    visual_occlusion_dark_pixel_threshold: int = 55
    visual_occlusion_dark_lower_ratio: float = 0.45
    visual_occlusion_dark_top_ratio: float = 0.60
    visual_occlusion_min_skin_ratio: float = 0.12
    visual_occlusion_skin_cr_min: int = 133
    visual_occlusion_skin_cr_max: int = 173
    visual_occlusion_skin_cb_min: int = 77
    visual_occlusion_skin_cb_max: int = 127
    visual_occlusion_det_score_margin: float = 0.08
    visual_occlusion_similarity_gap: float = 0.22
    visual_occlusion_min_candidate_similarity: float = 0.35
    camera_1_id: str = "entrada"
    camera_1_stream_url: str = "rtsp://iris-go2rtc:8554/usb_camera"
    camera_1_public_stream_url: str = ""
    camera_1_enabled: bool = True
    camera_1_source_kind: str = ""
    camera_1_device: str = ""
    camera_1_serial: str = ""
    camera_1_model: str = ""
    camera_1_stable_path: str = ""
    camera_2_id: str = "entrada_2"
    camera_2_stream_url: str = ""
    camera_2_public_stream_url: str = ""
    camera_2_enabled: bool = False
    camera_2_source_kind: str = ""
    camera_2_device: str = ""
    camera_2_serial: str = ""
    camera_2_model: str = ""
    camera_2_stable_path: str = ""
    camera_3_id: str = "entrada_3"
    camera_3_stream_url: str = ""
    camera_3_public_stream_url: str = ""
    camera_3_enabled: bool = False
    camera_3_source_kind: str = ""
    camera_3_device: str = ""
    camera_4_id: str = "entrada_4"
    camera_4_stream_url: str = ""
    camera_4_public_stream_url: str = ""
    camera_4_enabled: bool = False
    camera_4_source_kind: str = ""
    camera_4_device: str = ""
    usb_camera_device: str = "/dev/video0"
    usb_camera_input_format: str = "mjpeg"
    usb_camera_width: int = 1920
    usb_camera_height: int = 1080
    usb_camera_fps: int = 15
    onix_push_enabled: bool = False
    onix_push_url: str = ""
    onix_push_token: str = ""
    onix_push_device_uid: str = ""
    onix_push_timeout_seconds: float = 1.5
    onix_push_queue_size: int = 200
    onix_push_workers: int = 4
    onix_push_max_retries: int = 3
    onix_push_retry_delay_seconds: float = 1.0
    mqtt_publish_enabled: bool = False
    mqtt_publish_host: str = ""
    mqtt_publish_port: int = 1883
    mqtt_publish_username: str = ""
    mqtt_publish_password: str = ""
    mqtt_publish_client_id: str = ""
    mqtt_publish_topic_prefix: str = "iris"
    mqtt_publish_qos: int = 1
    mqtt_publish_queue_size: int = 500
    mqtt_publish_keepalive_seconds: int = 30
    mqtt_debounce_seconds: float = 30.0

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

    @property
    def effective_checks_per_second(self) -> float:
        interval = float(self.stream_capture_interval_seconds)
        return round(1.0 / interval, 3) if interval > 0 else 0.0


settings = Settings()
