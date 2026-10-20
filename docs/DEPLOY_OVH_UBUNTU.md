# Déploiement OVHcloud / Ubuntu 26.04

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

Le script installe Apache, Certbot et les outils de compilation, sans demander
`python3.12`, `python3.12-venv` ou `python3.12-dev` à APT et sans remplacer
`/usr/bin/python3`. Il détecte `uv` avec `command -v`, ou
`/home/deepforma/.local/bin/uv`, puis gère Python 3.12 avec `uv python install`,
`uv venv --python 3.12` et `uv pip install`. Si `.venv/bin/python` est déjà un
Python 3.12 fonctionnel, il est conservé; un venv d’une autre version n’est
jamais supprimé automatiquement. Le script n’exécute pas d’installateur distant
implicite pour uv.

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
Git, uv et Python sont exécutés sous `deepforma`; APT, systemd et Apache sous
root. `.env` est préservé et aucun `chmod 777` n’est utilisé.

## Apache et HTTPS

Le VirtualHost dédié à `deepforma.hephaestos.eu` proxyfie vers
`http://127.0.0.1:8001/`; le certificat HTTPS existant n’est pas écrasé.
`apache2ctl configtest` doit réussir avant tout reload. Après vérification DNS,
ports 80/443 et HTTP:

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
forward, vérifie `wsgi:application` et l’application, puis revient au commit
précédent si nécessaire. Gunicorn utilise un worker sur `127.0.0.1:8001`, avec
un timeout `GUNICORN_TIMEOUT` de 300 secondes par défaut et une limite systemd
de trois redémarrages en cinq minutes.
`deploy-prepare-data` est le seul raccourci de préparation CPF et vérifie les
entrées sans supprimer les datasets existants.

## Erreurs courantes

- `No module named common`: le service définit `PYTHONPATH=/opt/deepforma/app/src`.
- Gunicorn: l’entrée validée est `wsgi:application`, pas `web_app:app`.
- `Proxy Error`: consulter `make deploy-status`, `make deploy-logs` et
  `curl http://127.0.0.1:8001/health`.
- `PermissionError`: vérifier le groupe `deepforma` et les droits de
  `data/processed/cpf` avec `make deploy-check`.

Le mode sans modification est disponible pour les scripts concernés:

```bash
sudo bash deploy/install.sh --dry-run
sudo bash deploy/check.sh --dry-run
bash -n deploy/*.sh
python3 -m pytest -q tests/test_deployment.py tests/test_deploy_automation.py
```

Pour une installation initiale, préparez la clé SSH de `deepforma` et son
`known_hosts` pour GitHub; ne désactivez pas la vérification SSH. Le rollback ne
supprime ni les données CPF, ni les modèles, ni `.env`.
