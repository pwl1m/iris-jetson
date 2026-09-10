# iris-mosquitto

Broker MQTT local disponivel para o `iris-app` e integracoes locais. A
publicacao pelo Iris fica desligada por padrao; para usa-la localmente, configure
`MQTT_PUBLISH_ENABLED=true` e `MQTT_PUBLISH_HOST=iris-mosquitto` no `.env`.

## Conteudo

- `mosquitto.conf`: configuracao do broker.
- `data/`: persistencia local do Mosquitto.

## Nota De Migracao

Se ainda existir `data/mosquitto/`, mova o banco com sudo para preservar a persistencia criada pelo container:

```bash
sudo mv data/mosquitto/mosquitto.db iris-mosquitto/data/
sudo rmdir data/mosquitto data
```
