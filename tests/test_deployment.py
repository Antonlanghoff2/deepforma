from pathlib import Path
import subprocess

from src.web_app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_artifacts_exist():
    assert (ROOT / 'deploy' / 'deepforma.env.example').exists()
    assert (ROOT / 'deploy' / 'systemd' / 'deepforma.service').exists()
    assert (ROOT / 'deploy' / 'nginx' / 'deepforma.conf').exists()
    assert (ROOT / 'docs' / 'deployment_ubuntu.md').exists()


def test_systemd_unit_uses_real_entrypoint():
    text = (ROOT / 'deploy' / 'systemd' / 'deepforma.service').read_text(encoding='utf-8')
    assert 'ExecStart=/opt/deepforma/.venv/bin/gunicorn' in text
    assert 'src.web_app:create_app()' in text
    assert '127.0.0.1:8001' in text


def test_nginx_config_proxies_to_local_gunicorn():
    text = (ROOT / 'deploy' / 'nginx' / 'deepforma.conf').read_text(encoding='utf-8')
    assert 'proxy_pass http://127.0.0.1:8001;' in text
    assert 'proxy_set_header Host $host;' in text
    assert 'client_max_body_size 50M;' in text
    assert 'location /static/' in text


def test_env_example_contains_used_variables():
    text = (ROOT / 'deploy' / 'deepforma.env.example').read_text(encoding='utf-8')
    for key in [
        'FRANCE_TRAVAIL_CLIENT_ID',
        'FRANCE_TRAVAIL_CLIENT_SECRET',
        'FRANCE_TRAVAIL_SCOPE',
        'FRANCE_TRAVAIL_TOKEN_URL',
        'FRANCE_TRAVAIL_API_BASE_URL',
        'DEEPFORMA_CACHE_TTL_SECONDS',
        'HF_HOME',
        'TRANSFORMERS_CACHE',
    ]:
        assert key in text


def test_health_route_available():
    app = create_app()
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json()['status'] in {'ok', 'degraded'}


def test_shell_scripts_have_valid_syntax():
    for script in [
        ROOT / 'scripts' / 'deploy_ubuntu.sh',
        ROOT / 'scripts' / 'update_production.sh',
        ROOT / 'scripts' / 'rollback_production.sh',
    ]:
        subprocess.run(['bash', '-n', str(script)], check=True, cwd=ROOT)
