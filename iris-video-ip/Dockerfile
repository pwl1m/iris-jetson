# Validate this base against the Jetson host before building/running.
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:7.1-samples-multiarch
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 GST_DEBUG=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-gi python3-gst-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/iris-video-ip
COPY engine.py ./
EXPOSE 8090
ENTRYPOINT ["python3", "engine.py"]
