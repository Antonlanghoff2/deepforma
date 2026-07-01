from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from data.cpf_loader import (
    DEFAULT_SOURCE_CANDIDATES,
    inspect_cpf_source,
    prepare_cpf_v3_dataset,
    resolve_cpf_source,
)
from scripts.extract_cpf_skills import extract_cpf_skills


V3_COLUMNS = [
    '#',
    'Secteur',
    'Organisme de formation',
    'Intitulé de la formation',
    'Type de certification',
    'Code certification',
    'Niveau',
    'Codes ROME',
    'Compétences majeures',
    'Modalité',
    'Durée',
    'Prix TTC (€)',
    'Tags',
    '✅ Relu / Validé (oui/non)',
    '🗒 Corrections / Remarques',
]


def _write_v3_workbook(path: Path, rows: list[dict[str, object]], sheet_name: str = 'Generaliste_CPF') -> None:
    frame = pd.DataFrame.from_records(rows, columns=V3_COLUMNS)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _sample_v3_rows() -> list[dict[str, object]]:
    return [
        {
            '#': 1,
            'Secteur': 'Numérique',
            'Organisme de formation': 'Organisme Alpha',
            'Intitulé de la formation': 'Développeur Python',
            'Type de certification': 'RNCP',
            'Code certification': 'RNCP12345',
            'Niveau': '5',
            'Codes ROME': 'M1805 | M1806',
            'Compétences majeures': 'Python; Analyse de données | SQL',
            'Modalité': 'À distance',
            'Durée': '120 h',
            'Prix TTC (€)': '1200',
            'Tags': 'data ; python ; analyse de données',
            '✅ Relu / Validé (oui/non)': 'oui',
            '🗒 Corrections / Remarques': '',
        },
        {
            '#': 2,
            'Secteur': 'Numérique',
            'Organisme de formation': 'Organisme Alpha',
            'Intitulé de la formation': 'Développeur Python',
            'Type de certification': 'RNCP',
            'Code certification': 'RNCP12345',
            'Niveau': '5',
            'Codes ROME': 'M1805 | M1806',
            'Compétences majeures': 'Python; Analyse de données | SQL',
            'Modalité': 'À distance',
            'Durée': '120 h',
            'Prix TTC (€)': '1200',
            'Tags': 'data ; python ; analyse de données',
            '✅ Relu / Validé (oui/non)': 'oui',
            '🗒 Corrections / Remarques': '',
        },
        {
            '#': 3,
            'Secteur': 'Bureautique',
            'Organisme de formation': 'Organisme Beta',
            'Intitulé de la formation': '',
            'Type de certification': 'RS',
            'Code certification': 'RS999',
            'Niveau': '3',
            'Codes ROME': '',
            'Compétences majeures': '',
            'Modalité': 'Présentiel',
            'Durée': '35 h',
            'Prix TTC (€)': '350',
            'Tags': 'Excel; reporting',
            '✅ Relu / Validé (oui/non)': 'non',
            '🗒 Corrections / Remarques': 'Titre manquant',
        },
    ]


def test_detect_v3_source_and_fallback(tmp_path, monkeypatch):
    v2_path = tmp_path / 'Dataset_Generaliste_CPF_V2.xlsx'
    v1_path = tmp_path / 'Dataset_Generaliste_CPF_V1.xlsx'
    _write_v3_workbook(v2_path, _sample_v3_rows()[:1])
    _write_v3_workbook(v1_path, _sample_v3_rows()[:1])

    monkeypatch.setattr(
        'data.cpf_loader.DEFAULT_SOURCE_CANDIDATES',
        [tmp_path / 'missing.xlsx', v2_path, v1_path],
    )

    resolved = resolve_cpf_source(None)
    assert resolved == v2_path


def test_inspect_and_prepare_v3_dataset(tmp_path):
    source_path = tmp_path / 'Dataset_Generaliste_CPF_V3.xlsx'
    _write_v3_workbook(source_path, _sample_v3_rows())

    inspection = inspect_cpf_source(source_path)
    assert inspection.selected_sheet == 'Generaliste_CPF'
    assert inspection.row_count == 3
    assert inspection.column_count == len(V3_COLUMNS)
    assert inspection.non_null_by_column['Intitulé de la formation'] == 2

    prepared = prepare_cpf_v3_dataset(source_path, tmp_path / 'processed')
    frame = prepared.frame
    assert not frame.empty
    assert {'formation_id', 'titre', 'texte_modele', 'competences', 'competences_normalisees'}.issubset(frame.columns)
    assert frame['titre'].fillna('').astype(str).str.strip().ne('').all()
    assert frame['texte_modele'].fillna('').astype(str).str.strip().ne('').all()
    assert frame['competences'].apply(lambda value: bool(value)).any()
    assert any('Titre:' in text for text in frame['texte_modele'].tolist())
    assert any('Python' in str(value) for value in frame['competences'].tolist())

    normalized_path = tmp_path / 'processed' / 'cpf' / 'formations_normalized.parquet'
    enriched_path = tmp_path / 'processed' / 'cpf' / 'formations_with_skills.parquet'
    normalized = pd.read_parquet(normalized_path)
    enriched = extract_cpf_skills(normalized_path, enriched_path)

    assert normalized_path.exists()
    assert enriched_path.exists()
    assert len(normalized) == len(frame)
    assert len(enriched) == len(frame)
    assert enriched['skills_normalized'].apply(bool).any()
    assert enriched['skills_explicit'].apply(bool).any()


def test_prepare_v3_dataset_handles_empty_sheet_and_missing_title(tmp_path):
    empty_path = tmp_path / 'empty.xlsx'
    _write_v3_workbook(empty_path, [])
    inspection = inspect_cpf_source(empty_path)
    assert inspection.row_count == 0

    titleless_path = tmp_path / 'titleless.xlsx'
    titleless_row = dict(_sample_v3_rows()[0], **{'Intitulé de la formation': ''})
    good_row = dict(_sample_v3_rows()[1], **{'Code certification': 'RNCP99999', 'Intitulé de la formation': 'Analyse de données'})
    _write_v3_workbook(titleless_path, [titleless_row, good_row])

    prepared = prepare_cpf_v3_dataset(titleless_path, tmp_path / 'out')
    assert len(prepared.frame) == 1
    assert prepared.frame['titre'].fillna('').astype(str).str.strip().ne('').all()


def test_prepare_v3_dataset_reports_missing_source(tmp_path, monkeypatch):
    missing = tmp_path / 'missing.xlsx'
    monkeypatch.setattr('data.cpf_loader.DEFAULT_SOURCE_CANDIDATES', [tmp_path / 'fallback1.xlsx', tmp_path / 'fallback2.xlsx'])
    with pytest.raises(FileNotFoundError) as exc:
        resolve_cpf_source(missing)
    assert 'Aucun catalogue CPF trouvé' in str(exc.value)
    assert str(missing) in str(exc.value)


def test_makefile_pipeline_dependency_order_and_custom_variables(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'Dataset_Generaliste_CPF_V3.xlsx'
    _write_v3_workbook(source, _sample_v3_rows()[:1])
    output_dir = tmp_path / 'model'
    result = subprocess.run(
        [
            'make',
            '-n',
            'cpf-v3-all',
            f'CPF_SOURCE_FILE={source}',
            f'CPF_FORMATIONS={tmp_path / "processed" / "cpf" / "formations_with_skills.parquet"}',
            'CPF_BASE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            f'CPF_MODEL_OUTPUT={output_dir}',
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert 'scripts/inspect_cpf_dataset.py' in stdout
    assert 'scripts/prepare_cpf_dataset.py' in stdout
    assert 'scripts/extract_cpf_skills.py' in stdout
    assert 'scripts/build_cpf_training_pairs.py' in stdout
    assert 'scripts/train_cpf_recommender.py' in stdout
    assert 'scripts/evaluate_cpf_recommender.py' in stdout
    assert 'scripts/build_cpf_embeddings.py' in stdout
    assert str(source) in stdout
    assert str(output_dir) in stdout


@pytest.mark.parametrize(
    'script_name',
    [
        'inspect_cpf_dataset.py',
        'prepare_cpf_dataset.py',
    ],
)
def test_v3_cli_help_commands(script_name):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / 'scripts' / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), '--help'],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert 'usage' in result.stdout.lower()
