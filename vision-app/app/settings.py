from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    face_db_path: str = "/data/faces/faces.sqlite3"
    face_model_name: str = "buffalo_s"
    face_model_root: str = "/models/face"
    face_det_size: str = "640,640"
    face_similarity_threshold: float = 0.55
    face_max_results: int = 5
    face_providers: str = "CPUExecutionProvider"

    mqtt_host: str = "mosquitto"
    mqtt_port: int = 1883
    mqtt_topic: str = "frigate/events"
    frigate_url: str = "http://frigate:5000"
    recognition_event_types: str = "end"
    worker_log_level: str = "INFO"
    event_log_path: str = "/data/events/recognitions.jsonl"

    @property
    def det_size_tuple(self) -> tuple[int, int]:
        width, height = self.face_det_size.split(",", 1)
        return int(width), int(height)

    @property
    def providers_list(self) -> list[str]:
        return [item.strip() for item in self.face_providers.split(",") if item.strip()]

    @property
    def recognition_event_types_set(self) -> set[str]:
        return {item.strip() for item in self.recognition_event_types.split(",") if item.strip()}


settings = Settings()

