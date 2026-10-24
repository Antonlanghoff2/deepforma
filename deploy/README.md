# Déployer DeepForma sur un VPS Ubuntu

Ce dossier fournit une installation reproductible avec Apache, Gunicorn sous
systemd, un venv Python et une vérification de santé. L'installation ne
télécharge ni les modèles ni les datasets.

## Préparer le VPS

Utiliser Ubuntu 24.04 LTS ou plus récent, avec au moins 4 vCPU et 8 Go de RAM.
Créer le DNS `A` du domaine vers le VPS et utiliser un compte sudo.

Pour un dépôt privé, créer une clé de déploiement pour l'utilisateur applicatif
avant l'installation :

```bash
sudo groupadd --system deepforma 2>/dev/null || true
sudo useradd --system --create-home --home-dir /home/deepforma --gid deepforma deepforma 2>/dev/null || true
sudo -u deepforma ssh-keygen -t ed25519 -f /home/deepforma/.ssh/id_ed25519 -N ''
sudo cat /home/deepforma/.ssh/id_ed25519.pub
```

Ajouter la clé publique au dépôt GitHub, puis vérifier :

```bash
sudo -u deepforma ssh -o StrictHostKeyChecking=accept-new -T git@github.com
```

## Installer

Depuis le dépôt :

```bash
sudo REPO_URL=git@github.com:Antonlanghoff2/deepforma.git \
  REPO_BRANCH=main DOMAIN=app.example.com bash deploy/install.sh
```

Pour un dépôt public, remplacer `REPO_URL` par son URL HTTPS. Le script
installe les paquets système, crée l'utilisateur `deepforma`, clone le dépôt
dans `/opt/deepforma/app`, crée `.venv`, installe les dépendances, configure
Apache et démarre `deepforma.service`. `.env` est créé une seule fois et
préservé lors des mises à jour.

Tester sans modifier le serveur :

```bash
sudo REPO_URL=... DOMAIN=app.example.com bash deploy/install.sh --dry-run
```

## Configurer secrets et modèles

```bash
sudo -u deepforma nano /opt/deepforma/app/.env
sudo chmod 600 /opt/deepforma/app/.env
```

Renseigner les secrets France Travail et remplacer le mot de passe d'exemple.
Copier ensuite `models/binary_ia_v2` et `models/multilabel_competences_v2` dans
`/opt/deepforma/app/models/`, puis exécuter :

```bash
sudo chown -R deepforma:deepforma /opt/deepforma/app/models
sudo bash deploy/check.sh
```

Chaque modèle doit avoir `config.json`, `model.safetensors` et
`tokenizer_config.json` dans `final/`.

## HTTPS

Après propagation DNS et ouverture des ports 80/443 :

```bash
sudo ENABLE_CERTBOT=true SSL_EMAIL=admin@example.com \
  DOMAIN=app.example.com bash deploy/install.sh
```

## Exploitation

```bash
sudo bash deploy/check.sh
sudo bash deploy/update.sh
sudo bash deploy/restart.sh
sudo systemctl status deepforma --no-pager
sudo journalctl -u deepforma -f
sudo bash deploy/backup.sh
sudo bash deploy/rollback.sh
```

`deploy-update` refuse les modifications Git locales, sauvegarde avant mise à
jour et revient au commit précédent si les contrôles échouent. Le rollback ne
supprime ni `.env`, ni les données, ni les modèles.

Diagnostic rapide :

```bash
curl -i http://127.0.0.1:8001/health
sudo apache2ctl configtest
sudo journalctl -u deepforma -n 100 --no-pager
```

Les scripts ne lancent aucun entraînement. Sauvegarder aussi les données et les
archives de `/var/backups/deepforma` hors du VPS.
