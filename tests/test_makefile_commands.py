from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / 'Makefile'
README = REPO_ROOT / 'README.md'


def extract_make_targets(text: str) -> set[str]:
    targets: set[str] = set()
    for line in text.splitlines():
        if line.startswith(('\t', ' ')):
            continue
        if ':' not in line or line.lstrip().startswith('#'):
            continue
        left = line.split(':', 1)[0].strip()
        if left and '=' not in left and ' ' not in left:
            targets.add(left)
    return targets


def extract_readme_make_commands(text: str) -> set[str]:
    commands: set[str] = set()
    for match in re.finditer(r'make\s+([A-Za-z0-9_.-]+)', text):
        commands.add(match.group(1))
    return commands


def test_readme_make_commands_exist_in_makefile() -> None:
    make_targets = extract_make_targets(MAKEFILE.read_text(encoding='utf-8'))
    readme_commands = extract_readme_make_commands(README.read_text(encoding='utf-8'))
    missing = sorted(cmd for cmd in readme_commands if cmd not in make_targets)
    assert missing == []


def test_key_make_targets_and_aliases_exist() -> None:
    text = MAKEFILE.read_text(encoding='utf-8')
    expected = [
        'help',
        'install',
        'setup',
        'run',
        'dev',
        'clean',
        'test',
        'smoke-test',
        'binary-ai-prepare',
        'binary-ai-train-ml',
        'binary-ai-train-dl',
        'binary-ai-compare',
        'binary-ai-evaluate',
        'binary-ai-all',
        'cpf-general-check',
        'cpf-general-prepare',
        'cpf-pairs',
        'cpf-train',
        'cpf-general-all',
        'cpf-all',
        'ia-check',
        'ia-prepare',
        'ia-train',
        'ia-evaluate',
        'ia-all',
        'france-travail-check',
        'france-travail-collect',
        'model-check',
    ]
    for target in expected:
        assert re.search(rf'^{re.escape(target)}:', text, re.M), target
    assert re.search(r'^cpf-all:\s+cpf-general-all\b', text, re.M)
    assert re.search(r'^binary-ai-evaluate:\s+binary-ai-compare\b', text, re.M)
    assert re.search(r'^ia-all:\s+ia-evaluate\b', text, re.M)


def test_main_pipeline_dry_runs_parse() -> None:
    subprocess.run(['make', '-n', 'cpf-general-all'], cwd=REPO_ROOT, check=True)
    subprocess.run(['make', '-n', 'ia-all'], cwd=REPO_ROOT, check=True)
    subprocess.run(['make', '-n', 'binary-ai-all'], cwd=REPO_ROOT, check=True)
