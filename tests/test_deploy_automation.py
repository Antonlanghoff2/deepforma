from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_scripts_have_valid_bash_syntax():
    for script in sorted((ROOT / "deploy").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True, cwd=ROOT)


def test_wsgi_exposes_wsgi_application():
    source = (ROOT / "src/wsgi.py").read_text(encoding="utf-8")
    assert "application = create_app(" in source
    assert "from web_app import create_app" in source


def test_make_overlay_exposes_requested_commands():
    text = (ROOT / "GNUmakefile").read_text(encoding="utf-8")
    for target in ("deploy", "deploy-check", "deploy-update", "deploy-restart", "deploy-status", "deploy-logs", "deploy-backup", "deploy-rollback", "deploy-prepare-data"):
        assert f"{target}:" in text or f"{target} deploy-install:" in text


def test_restart_dry_run_does_not_require_root():
    result = subprocess.run(["bash", "deploy/restart.sh", "--dry-run"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0 and "systemctl" in result.stdout
