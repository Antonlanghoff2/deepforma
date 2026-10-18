# Déploiement Deepforma

Les scripts ciblent Ubuntu, Apache 2.4, Python 3.12 et Gunicorn sur
`127.0.0.1:8001`. Ils ne lancent pas les pipelines métier et ne téléchargent
pas de modèles.

```bash
sudo bash deploy/install.sh
make deploy-check
make deploy-update
make deploy-restart
make deploy-status
make deploy-logs
make deploy-backup
make deploy-rollback
```

`--dry-run` est accepté par les scripts modificateurs. `.env` est préservé et
doit rester en `0600`. Les modèles historiques sont transférés séparément dans
`DEEPFORMA_MODELS_DIR` (défaut `/opt/deepforma/app/models`). Le rollback ne
supprime jamais les données CPF ni les poids de modèles.
