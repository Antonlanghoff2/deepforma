\
# Déploiement Ubuntu Deepforma

Ce guide décrit un déploiement reproductible sur Ubuntu 22.04 ou 24.04 avec Git, Python 3, `.venv`, Gunicorn, systemd et Nginx.

## Pré-requis serveur

- Ubuntu 22.04 ou 24.04
- `git`
- `python3`
- `python3-venv`
- `python3-pip`
- `python3-dev`
- `build-essential`
- `nginx`
- `curl`
- `ca-certificates`
- `openssh-client`

Le projet ne dépend pas d'une base de données relationnelle. Aucun `DATABASE_URL` n'est requis par le code actuel.

## Utilisateur système

Le script de déploiement crée un utilisateur système dédié `deepforma` si nécessaire.

En production, ce compte ne devrait pas disposer de privilèges sudo permanents. Si une intervention manuelle est nécessaire, on peut ponctuellement l'ajouter au groupe sudo :

```bash
sudo usermod -aG sudo deepforma
```

Une nouvelle connexion est nécessaire pour que ce changement soit pris en compte.

## Point d'entrée application

L'application web Deepforma est une application Flask avec une factory :

- module : `src.web_app`
- factory Gunicorn : `src.web_app:create_app()`
- route de santé : `GET /health`

Gunicorn est lié à `127.0.0.1:8001`.

## Variables d'environnement

Le fichier de production `.env` ne doit pas être versionné. Un exemple est fourni dans `deploy/deepforma.env.example`.

Variables actuellement utilisées par le code :

- `PYTHONUNBUFFERED`
- `FRANCE_TRAVAIL_CLIENT_ID`
- `FRANCE_TRAVAIL_CLIENT_SECRET`
- `FRANCE_TRAVAIL_SCOPE`
- `FRANCE_TRAVAIL_TOKEN_URL`
- `FRANCE_TRAVAIL_API_BASE_URL`
- `FRANCE_TRAVAIL_TIMEOUT`
- `DEEPFORMA_CACHE_TTL_SECONDS`
- `DEEPFORMA_MAX_OFFERS`
- `DEEPFORMA_PAGE_SIZE`
- `DEEPFORMA_MAX_PAGES`
- `DEEPFORMA_DEFAULT_THRESHOLD`
- `DEEPFORMA_MAX_LENGTH`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `TOKENIZERS_PARALLELISM`

Variables non utilisées par le code actuel et donc non imposées ici : `SECRET_KEY`, `DATABASE_URL`, `MODEL_PATH`, `CPF_MODEL_PATH`, `IA_MODEL_PATH`.

## Premier déploiement

Depuis le dépôt cloné localement :

```bash
sudo APP_USER=deepforma \
  APP_GROUP=deepforma \
  APP_DIR=/opt/deepforma \
  REPO_URL=git@github.com:Antonlanghoff2/deepforma.git \
  REPO_BRANCH=main \
  DOMAIN=deepforma.hephaestos.eu \
  ENABLE_SSL=true \
  SSL_EMAIL=antonlanghoff@gmail.com \
  bash scripts/deploy_ubuntu.sh
```

### Mode simulation

```bash
DRY_RUN=true bash scripts/deploy_ubuntu.sh
```

## Mise à jour

```bash
sudo APP_USER=deepforma \
  APP_GROUP=deepforma \
  APP_DIR=/opt/deepforma \
  REPO_URL=git@github.com:Antonlanghoff2/deepforma.git \
  REPO_BRANCH=main \
  bash scripts/update_production.sh
```

## Logs et statut

```bash
sudo systemctl status deepforma
sudo journalctl -u deepforma -f
```

## Redémarrage

```bash
sudo systemctl restart deepforma
```

## Rollback

Par hash :

```bash
sudo bash scripts/rollback_production.sh --commit <SHA>
```

Ou avec le dernier hash enregistré par la mise à jour :

```bash
sudo bash scripts/rollback_production.sh
```

## HTTPS

Si `ENABLE_SSL=true`, le script tente l'installation de Certbot puis la configuration du certificat via Nginx.

Points à vérifier :

- le domaine doit pointer vers le serveur
- `SSL_EMAIL` doit être renseigné
- le certificat peut être installé après validation de la configuration Nginx
- si Certbot échoue, le service HTTP reste disponible

## Permissions

Les répertoires persistants doivent rester hors de Git :

- `/opt/deepforma/data`
- `/opt/deepforma/models`
- `/opt/deepforma/logs`
- `/opt/deepforma/.cache/huggingface`

Le fichier `/opt/deepforma/.env` doit appartenir à `deepforma:deepforma` et être en `chmod 600`.

## Modèles IA

Les modèles lourds ne sont pas téléchargés automatiquement pendant le déploiement, sauf si un script dédié du projet l'exige explicitement.

## Erreurs fréquentes

- clé SSH GitHub absente pour l'utilisateur `deepforma`
- dépôt local modifié avant pull `--ff-only`
- variables France Travail manquantes lors de l'analyse marché
- DNS du domaine non propagé avant Certbot
- Nginx en erreur de configuration
