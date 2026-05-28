# iris-mosquitto

Broker MQTT local usado pelo `iris-app` e por integracoes locais.

## Conteudo

- `mosquitto.conf`: configuracao do broker.
- `data/`: persistencia local do Mosquitto.

## Nota De Migracao

Se ainda existir `data/mosquitto/`, mova o banco com sudo para preservar a persistencia criada pelo container:

```bash
sudo mv data/mosquitto/mosquitto.db iris-mosquitto/data/
sudo rmdir data/mosquitto data
```
