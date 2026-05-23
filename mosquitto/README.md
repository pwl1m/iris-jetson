# Mosquitto

Broker MQTT local usado pelo Frigate e pelo `vision-app`.

## Conteudo

- `mosquitto.conf`: configuracao do broker.
- `data/`: persistencia local do Mosquitto.

## Nota De Migracao

Se ainda existir `data/mosquitto/`, mova o banco com sudo para preservar a persistencia criada pelo container:

```bash
sudo mv data/mosquitto/mosquitto.db mosquitto/data/
sudo rmdir data/mosquitto data
```
