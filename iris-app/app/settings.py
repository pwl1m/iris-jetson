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
    # Contagem de pessoas por frame (censo persistido, distinto do /crowd em
    # memoria). Arquivo proprio, fora de faces.sqlite3, porque nao e dado
    # biometrico: e so "quantos rostos apareceram", sem embedding nem
    # identidade. Ver PEOPLE_COUNT_API.md (raiz de iris-app/).
    frame_census_db_path: str = "/data/events/census.sqlite3"
    camera_name: str = "entrada"
    stream_source_kind: str = "rtsp"
    stream_url: str = "rtsp://iris-go2rtc:8554/usb_camera"
    stream_gst_pipeline: str = ""
    stream_worker_enabled: bool = True
    ip_engine_token_file: str = "/run/secrets/engine_token"
    ip_engine_max_frame_age_seconds: float = 2.0
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
    pipeline_single_face: bool = False
    # Tres orcamentos distintos, porque detectar, rastrear e materializar custam
    # coisas diferentes.  Medido nesta Jetson com det_size 960: detectar custa
    # ~43 ms por frame, plano, pedindo 3 ou 50 rostos -- o SCRFD ja varre o frame
    # inteiro e max_faces so corta a lista depois.  Materializar (landmarks +
    # embedding) custa ~21 ms por rosto e e o unico que escala.
    #
    # pipeline_detect_max_faces: censo do frame.  Barato, serve para contar quem
    #   passa mesmo sem ter angulo ou tamanho para reconhecer.
    # pipeline_max_faces: quantos tracks simultaneos o tracker mantem.  Custa
    #   memoria e casamento IoU, nao GPU.
    # pipeline_materialize_budget: quantos tracks viram landmarks+embedding POR
    #   FRAME.  Este e o que consome o orcamento: a 5 FPS sao 200 ms por frame,
    #   43 ms vao para deteccao e sobram ~150 ms, ou seja 5 rostos com folga
    #   (p95 medido de 167,8 ms) e 6 nao (221,6 ms).
    #
    # Como os tracks vivem ttl_seconds (1,25 s = ~6 frames a 5 FPS), um orcamento
    # de 5 por frame materializa ate ~30 tracks dentro de uma janela de TTL sem
    # estourar frame algum.
    pipeline_detect_max_faces: int = 20
    pipeline_max_faces: int = 5
    pipeline_materialize_budget: int = 5
    pipeline_track_iou_threshold: float = 0.25
    pipeline_track_ttl_seconds: float = 1.25
    pipeline_track_min_frames: int = 2
    pipeline_track_retry_seconds: float = 0.75
    pipeline_track_min_face_size: int = 32
    # Portoes de persistencia de evento. Cada evento gravado custa uma linha de
    # JSONL e DOIS JPEGs, porque `_save_capture` e `_save_face_crop` rodam antes
    # da escrita. Medido na linha USB, 3,5 meses de uma camera: 491.504 eventos,
    # ~992 MB de JSONL e ~3,3 GB de imagem.
    #
    # Dois desperdicios distintos aparecem nesses dados:
    #   1. Gente parada. Jose 101.489 eventos e Emanuel 52.579 -- juntos 98% de
    #      tudo que foi identificado, duas pessoas sentadas na mesa.
    #   2. Rosto pequeno demais para servir. Dos 334.761 no_match, 84,8% tem
    #      48-64 px e det_score medíocre: nao sustentam veredito de oclusao nem
    #      embedding confiavel, e na fase de recorrencia seriam a origem das
    #      fusoes erradas de identidade.
    #
    # ZERO DESLIGA, e zero e o default DE PROPOSITO: o ensaio de campo precisa
    # gravar tudo, porque e dele que sai o rotulo para calibrar. Alem disso os
    # numeros acima sao da camera USB, com outra luz e outro enquadramento.
    # Ligar depois do ensaio, com os cortes que o scene_baseline.py indicar.
    #
    # Oclusao nunca passa por estes portoes: sao 1.997 eventos no periodo todo,
    # alto valor por unidade.
    event_debounce_seconds: float = 0.0
    event_unmatched_min_width: int = 0
    event_unmatched_min_det_score: float = 0.0
    pipeline_warm_up_enabled: bool = True
    pipeline_save_face_crop: bool = True
    pipeline_face_crop_padding: float = 0.25
    face_min_det_score: float = 0.65
    face_min_width: int = 48
    face_min_height: int = 48
    face_min_blur_score: float = 40.0
    visual_occlusion_enabled: bool = True
    visual_occlusion_score_threshold: float = 0.75
    # Piso de AVALIACAO: abaixo disto nem as metricas sao calculadas. Alinhado
    # com face_min_width/height, porque um rosto grande o bastante para decidir
    # identidade e grande o bastante para ser medido. Com 96 a heuristica nao
    # rodava em 99,4% dos rostos reais da linha USB.
    visual_occlusion_min_width: int = 48
    visual_occlusion_min_height: int = 48
    # Piso de VEREDITO: abaixo disto as metricas sao registradas mas `suspected`
    # nunca fica True. Em rostos de 48-96px a regra atual marcaria 4,84% dos
    # rostos normais como oclusao, medido em 9156 crops reais. Mantido em 96 ate
    # o ensaio com mascara e oculos dar rotulo para recalibrar os limiares.
    # Piso de veredito: abaixo dele as metricas sao calculadas mas `suspected`
    # nunca fica True. Era 96 px, escolhido quando o unico dado disponivel vinha
    # da camera USB, onde 99,4% dos rostos ficavam abaixo desse piso e a regra
    # praticamente nunca agia.
    #
    # Medido no cenario IP real em 21/09/2026, com pessoas passando: a
    # distribuicao mudou. Nenhum rosto abaixo de 48 px, 51,1% entre 48 e 64,
    # 23,4% entre 64 e 96 e 25,5% acima de 96. Baixar para 64 leva a cobertura
    # de 25,5% para cerca de 49% do trafego, e o cenario sustenta isso: o
    # det_score mediano subiu para 0,837, a nitidez para 1.170 e o skin_ratio
    # para 0,680, todos bem acima da linha USB, com ZERO falso positivo de
    # oclusao em 48 eventos.
    #
    # Nao foi para 48 de uma vez de proposito: a faixa 48-64 e metade do
    # trafego e ainda nao tem nenhum caso rotulado de oclusao real para validar.
    visual_occlusion_verdict_min_size: int = 64
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
    # Dica de como a URL publica deve ser tocada. `mjpeg` mantem exatamente o
    # comportamento de hoje. `hls` ou `fmp4` valem quando o go2rtc estiver no ar
    # e o patch de render_mode tiver sido aplicado no Onix.
    camera_stream_render_mode: str = "mjpeg"
    camera_1_public_stream_url: str = ""
    camera_1_lan_stream_url: str = ""
    camera_1_tailscale_stream_url: str = ""
    camera_1_enabled: bool = True
    camera_1_source_kind: str = ""
    camera_1_device: str = ""
    camera_1_serial: str = ""
    camera_1_model: str = ""
    camera_1_stable_path: str = ""
    # Zona que o censo de pessoas conta, em fracoes de 0 a 1 ("x1,y1,x2,y2"),
    # nunca em pixel absoluto -- pixel absoluto quebraria de novo na proxima
    # troca de resolucao da camera (ja aconteceu uma vez, D-019). Vazio
    # desliga o filtro e o censo conta o frame inteiro, que e o default.
    # Filtra so `/people-count` e `frame_census`; recognition/tracking/oclusao
    # continuam vendo o frame inteiro sem mudanca de comportamento.
    camera_1_roi: str = ""
    camera_2_id: str = "entrada_2"
    camera_2_stream_url: str = ""
    camera_2_public_stream_url: str = ""
    camera_2_lan_stream_url: str = ""
    camera_2_tailscale_stream_url: str = ""
    camera_2_enabled: bool = False
    camera_2_source_kind: str = ""
    camera_2_device: str = ""
    camera_2_serial: str = ""
    camera_2_model: str = ""
    camera_2_stable_path: str = ""
    camera_2_roi: str = ""
    camera_3_id: str = "entrada_3"
    camera_3_stream_url: str = ""
    camera_3_public_stream_url: str = ""
    camera_3_lan_stream_url: str = ""
    camera_3_tailscale_stream_url: str = ""
    camera_3_enabled: bool = False
    camera_3_source_kind: str = ""
    camera_3_device: str = ""
    camera_3_roi: str = ""
    camera_4_id: str = "entrada_4"
    camera_4_stream_url: str = ""
    camera_4_public_stream_url: str = ""
    camera_4_lan_stream_url: str = ""
    camera_4_tailscale_stream_url: str = ""
    camera_4_enabled: bool = False
    camera_4_source_kind: str = ""
    camera_4_device: str = ""
    camera_4_roi: str = ""
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
    # Enriquecimento visual opcional. A inferencia nunca roda na thread de
    # reconhecimento: esta fila apenas registra o evento no orquestrador local.
    vlm_enrichment_enabled: bool = False
    vlm_orchestrator_url: str = "http://eclusa-vlm-orchestrator:18100/v1/jobs"
    vlm_api_token: str = ""
    vlm_api_token_file: str = ""
    vlm_prompt_id: str = "iris-scene-v1"
    vlm_priority: int = 100
    vlm_queue_size: int = 200
    vlm_submit_timeout_seconds: float = 2.0
    vlm_submit_max_retries: int = 3
    vlm_submit_retry_delay_seconds: float = 1.0
    vlm_job_max_attempts: int = 3
    # Mesmo volume de /data/events, montado read-only neste caminho no
    # orquestrador. Assim nenhuma imagem precisa ser copiada ou exposta em HTTP.
    vlm_shared_event_root: str = "/sources/iris-events"
    vlm_callback_url: str = ""
    api_allowed_client_ips: str = ""

    @property
    def effective_detect_max_faces(self) -> int:
        """O censo nunca pode ser menor que a capacidade do tracker."""
        return max(int(self.pipeline_detect_max_faces), int(self.pipeline_max_faces))

    @property
    def effective_materialize_budget(self) -> int:
        """Teto de rostos materializados por frame, limitado pela capacidade do tracker.

        Materializar mais do que o tracker consegue manter simultaneamente nao
        tem efeito, entao o orcamento e limitado a `pipeline_max_faces`.  O
        default de codigo ja e coerente (5 e 5); o clamp existe para que baixar
        so a capacidade do tracker nao deixe um orcamento pendurado maior que
        ela.  Zero ou negativo desliga o teto.
        """
        budget = int(self.pipeline_materialize_budget)
        if budget <= 0:
            return 0
        return min(budget, int(self.pipeline_max_faces))

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

    @property
    def api_allowed_clients(self) -> set[str]:
        return {item.strip() for item in self.api_allowed_client_ips.split(",") if item.strip()}


settings = Settings()
