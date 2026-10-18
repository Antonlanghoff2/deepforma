from pathlib import Path
import subprocess

from src.web_app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_artifacts_exist():
    for path in ("deploy/common.sh", "deploy/install.sh", "deploy/update.sh", "deploy/restart.sh", "deploy/check.sh", "deploy/backup.sh", "deploy/rollback.sh", "deploy/systemd/deepforma.service", "deploy/apache/deepforma.conf", "docs/DEPLOY_OVH_UBUNTU.md"):
        assert (ROOT / path).exists()


def test_systemd_unit_uses_real_entrypoint():
    text = (ROOT / "deploy/systemd/deepforma.service").read_text(encoding="utf-8")
    assert "ExecStart=/opt/deepforma/app/.venv/bin/gunicorn" in text
    assert "wsgi:application" in text and "127.0.0.1:8001" in text
    assert "User=deepforma" in text and "--workers 1" in text


def test_apache_config_proxies_to_local_gunicorn():
    text = (ROOT / "deploy/apache/deepforma.conf").read_text(encoding="utf-8")
    assert "ProxyPass / http://127.0.0.1:8001/" in text
    assert "ProxyPreserveHost On" in text and "LimitRequestBody 52428800" in text
    assert "Alias /static/ /opt/deepforma/app/static/" in text


def test_wsgi_import_contract():
    source = (ROOT / "src/wsgi.py").read_text(encoding="utf-8")
    assert "application = create_app(" in source


def test_health_route_available():
    response = create_app().test_client().get("/health")
    assert response.status_code == 200 and response.is_json
    assert response.get_json()["status"] in {"ok", "degraded"}


def test_shell_scripts_have_valid_syntax():
    for script in sorted((ROOT / "deploy").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True, cwd=ROOT)
