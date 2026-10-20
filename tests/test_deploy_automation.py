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
    overlay = (ROOT / "GNUmakefile").read_text(encoding="utf-8")
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "include Makefile" in overlay
    for target in ("deploy", "deploy-check", "deploy-update", "deploy-restart", "deploy-status", "deploy-logs", "deploy-backup", "deploy-rollback", "deploy-prepare-data"):
        assert f"{target}:" in text


def test_restart_dry_run_does_not_require_root():
    result = subprocess.run(["bash", "deploy/restart.sh", "--dry-run"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0 and "systemctl" in result.stdout


def test_deploy_configuration_does_not_install_system_python312():
    install = (ROOT / "deploy/install.sh").read_text(encoding="utf-8")
    assert "python3.12" not in install
    assert "python3.12-venv" not in install
    assert "python3.12-dev" not in install


def test_makefiles_have_one_recipe_per_deploy_target():
    result = subprocess.run(["make", "-pn", "deploy"], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "overriding recipe for target" not in result.stderr
    assert "ignoring old recipe for target" not in result.stderr


def test_check_dry_run_is_non_mutating():
    result = subprocess.run(["bash", "deploy/check.sh", "--dry-run"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0
    assert "aucune modification" in result.stdout
