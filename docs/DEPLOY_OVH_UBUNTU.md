# Déploiement OVHcloud / Ubuntu

Le serveur utilise `ubuntu` pour SSH/SFTP et `deepforma` pour l’application.
Le dépôt est cloné avec `git@github.com:Antonlanghoff2/deepforma.git`; gardez la
vérification SSH active.

## Installation

```bash
sudo install -d -o ubuntu -g ubuntu /opt/deepforma
cd /opt/deepforma
git clone git@github.com:Antonlanghoff2/deepforma.git app
cd app
sudo bash deploy/install.sh
```

Le script installe Apache, Certbot, Python 3.12 et les outils de compilation
sans remplacer `/usr/bin/python3`. Il crée `/opt/deepforma/app/.venv` et installe
`pip` si le venv créé par `uv` ne le contient pas.

## Permissions, secrets et modèles

Le groupe `deepforma` est utilisé pour les répertoires partagés; les répertoires
sont setgid et aucun `chmod 777` n’est utilisé. `.env` reste en `0600` et n’est
jamais écrasé. Transférez les modèles sans les committer:

```bash
sudo rsync -a models/binary_ia_v2 models/multilabel_competences_v2 /opt/deepforma/app/models/
sudo chown -R deepforma:deepforma /opt/deepforma/app/models
```

Le démarrage vérifie `config.json`, `model.safetensors` et le tokenizer des deux
répertoires historiques; aucun téléchargement ou réentraînement n’est lancé.

## Apache et HTTPS

Le VirtualHost dédié proxyfie seulement vers `127.0.0.1:8001`; `apache2ctl
configtest` doit réussir avant tout reload. Après vérification DNS, ports 80/443
et HTTP:

```bash
sudo ENABLE_CERTBOT=true SSL_EMAIL=admin@example.org bash deploy/install.sh
```

Certbot utilise `--keep-until-expiring`; son timer assure les renouvellements.

## Exploitation

```bash
make deploy-check
make deploy-update
make deploy-restart
make deploy-status
make deploy-logs
make deploy-backup
make deploy-rollback
make deploy-prepare-data
```

`deploy-update` refuse les changements locaux, sauvegarde, met à jour en fast
forward, vérifie l’application et revient au commit précédent si nécessaire.
`deploy-prepare-data` est le seul raccourci de préparation CPF et vérifie les
entrées sans supprimer les datasets existants.

## Erreurs courantes

- `No module named common`: le service définit `PYTHONPATH=/opt/deepforma/app/src`.
- Gunicorn: l’entrée validée est `wsgi:application`, pas `web_app:app`.
- `Proxy Error`: consulter `make deploy-status`, `make deploy-logs` et
  `curl http://127.0.0.1:8001/health`.
- `PermissionError`: vérifier le groupe `deepforma` et les droits de
  `data/processed/cpf` avec `make deploy-check`.
