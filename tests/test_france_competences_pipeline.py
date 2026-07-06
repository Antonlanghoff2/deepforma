from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from data_sources.france_competences.client import FranceCompetencesClient, FranceCompetencesVerificationError, safe_extract_zip
from data_sources.france_competences.rncp_parser import FranceCompetencesRncpParser
from data_sources.france_competences.skill_extractor import decompose_composite_skill
from scripts.build_france_competences_training_dataset import _split_for_certification


def _make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, 'w') as zf:
        for name, payload in files.items():
            zf.writestr(name, payload)
    return path


def test_select_latest_resource_prefers_latest_by_kind() -> None:
    client = FranceCompetencesClient(dataset_slug='demo')
    resources = [
        {'id': 'old-rncp', 'title': 'export-fiches-rncp-v4-1-2026-07-04.zip', 'format': 'zip', 'last_modified': '2026-07-04T02:01:00+00:00', 'filesize': 1, 'url': 'https://example.com/old-rncp.zip'},
        {'id': 'new-rncp', 'title': 'export-fiches-rncp-v4-1-2026-07-05.zip', 'format': 'zip', 'last_modified': '2026-07-05T02:01:00+00:00', 'filesize': 2, 'url': 'https://example.com/new-rncp.zip'},
        {'id': 'new-rs', 'title': 'export-fiches-rs-v4-1-2026-07-05.zip', 'format': 'zip', 'last_modified': '2026-07-05T02:01:00+00:00', 'filesize': 3, 'url': 'https://example.com/new-rs.zip'},
    ]
    rncp = client.select_latest_resource(resources, include_rncp=True, include_rs=False)
    rs = client.select_latest_resource(resources, include_rncp=False, include_rs=True)
    assert rncp['id'] == 'new-rncp'
    assert rs['id'] == 'new-rs'


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / 'malicious.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('../evil.txt', b'evil')
    with pytest.raises(FranceCompetencesVerificationError):
        safe_extract_zip(archive, tmp_path / 'extract')


def test_download_resource_is_atomic_and_verified(tmp_path: Path) -> None:
    client = FranceCompetencesClient(dataset_slug='demo')
    resource = {
        'id': 'res-1',
        'title': 'export-fiches-rncp-v4-1-2026-07-05.zip',
        'format': 'zip',
        'filesize': 5,
        'checksum': {'type': 'sha1', 'value': 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'},
        'url': 'https://example.com/export.zip',
    }

    def fake_stream(url: str, destination: Path) -> None:
        destination.write_bytes(b'hello')

    client._stream_download = fake_stream  # type: ignore[method-assign]
    target, verification = client.download_resource(resource, tmp_path)
    assert target.exists()
    assert verification.size == 5
    assert verification.ok is True
    assert target.read_bytes() == b'hello'
    assert not list(tmp_path.glob('*.download'))


def test_rncp_parser_extracts_bullets_and_excludes_evaluation(tmp_path: Path) -> None:
    xml = """<FICHES>
  <VERSION_FLUX>4.1</VERSION_FLUX>
  <FICHE>
    <ID_FICHE>1</ID_FICHE>
    <NUMERO_FICHE>RNCP1</NUMERO_FICHE>
    <INTITULE>Test</INTITULE>
    <ETAT_FICHE>Publiée</ETAT_FICHE>
    <NOMENCLATURE_EUROPE><NIVEAU>NIV7</NIVEAU><LIBELLE>Niveau 7</LIBELLE></NOMENCLATURE_EUROPE>
    <CODES_ROME><ROME><CODE>M1805</CODE></ROME></CODES_ROME>
    <CERTIFICATEURS><CERTIFICATEUR><NOM_CERTIFICATEUR>A</NOM_CERTIFICATEUR></CERTIFICATEUR></CERTIFICATEURS>
    <BLOCS_COMPETENCES>
      <BLOC_COMPETENCES>
        <CODE>RNCP1BC01</CODE>
        <LIBELLE>Bloc 1</LIBELLE>
        <LISTE_COMPETENCES>- Analyser les données\n- Déployer un modèle\nEvaluation : test</LISTE_COMPETENCES>
      </BLOC_COMPETENCES>
    </BLOCS_COMPETENCES>
  </FICHE>
</FICHES>"""
    path = tmp_path / 'rncp.xml'
    path.write_text(xml, encoding='utf-8')
    parsed = FranceCompetencesRncpParser().parse_archive(path)
    assert len(parsed.certifications) == 1
    assert len(parsed.blocks) == 1
    assert [skill['libelle_officiel'] for skill in parsed.skills] == ['Analyser les données', 'Déployer un modèle']
    assert all('test' not in skill['libelle_officiel'].lower() for skill in parsed.skills)


def test_composite_skill_decomposition() -> None:
    parts = decompose_composite_skill('Nettoyer, normaliser et encoder les données')
    assert any('nettoyer' in part.lower() for part in parts)
    assert len(parts) >= 3


def test_split_by_certification_is_stable() -> None:
    assert _split_for_certification('RNCP123') == _split_for_certification('RNCP123')
    assert _split_for_certification('RNCP123') in {'train', 'validation', 'test'}


def test_schema_change_and_empty_inputs(tmp_path: Path) -> None:
    empty = tmp_path / 'empty.zip'
    with zipfile.ZipFile(empty, 'w'):
        pass
    parsed = FranceCompetencesRncpParser().parse_archive(empty)
    assert parsed.certifications == []
    assert parsed.blocks == []
    assert parsed.skills == []

