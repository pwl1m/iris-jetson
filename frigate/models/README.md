# Detection Models

Este diretorio guarda engines TensorRT e artefatos de deteccao usados pelo Frigate/DeepStream.

Fase 1 Jetson:

- Frigate TensorRT JP6
- engine inicial esperada em `tensorrt/yolov7-320.trt`

Fase 2:

- PeopleNet 2.6 para deteccao de pessoa
- YOLOv8s/n para classes customizadas

Engines TensorRT devem ser geradas em ambiente compativel com o Jetson Orin Nano e a versao do JetPack.

