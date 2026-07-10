from __future__ import annotations

from pathlib import Path

from referential_import.competency_parser import parse_competency_and_criteria_lines
from referential_import.import_service import ReferentialImportService
from referential_import.models import OfficialCompetency
from referential_import.pdf_loader import PdfDocument, PdfPage, PdfTextBlock
from referential_import.skill_decomposer import decompose_competency
from referential_import.store import ReferentialImportStore
from referential_import.table_extractor import ExtractedCell, ExtractedTablePage


def test_parse_competency_and_criteria_skips_modalities():
    lines = [
        (1, "C1.1 Organiser la coordination et la coopération des différents intervenants internes et externes en utilisant la méthode du Lean Management pour améliorer la performance de l'entreprise"),
        (2, "Modalités d'évaluation : mise en situation professionnelle"),
        (3, "CE1.1.1 La coordination est structurée et explicable"),
        (4, "CE1.1.2 La performance est mesurée et partagée"),
    ]
    competencies, criteria, _ = parse_competency_and_criteria_lines(lines, page_number=13)
    assert len(competencies) == 1
    assert competencies[0].code == "C1.1"
    assert "Modalités" not in competencies[0].official_label
    assert len(criteria) == 2
    assert criteria[0].competency_code == "C1.1"
    assert criteria[1].competency_code == "C1.1"
    assert competencies[0].evaluation_criteria


def test_skill_decomposer_extracts_tools_methods_and_regulatory():
    competency = OfficialCompetency(
        code="C4.1",
        official_label="Organiser la coordination en utilisant Excel, le Lean Management, le PDCA et le RGPD.",
        normalized_label="organiser la coordination en utilisant excel le lean management le pdca et le rgpd",
        block_code="BLOC_4",
        activity_code="A4.1",
        page_start=13,
        page_end=13,
        confidence=0.98,
        source_pages=[13],
    )
    derived = decompose_competency(competency, [])
    labels = {item.canonical_label for item in derived}
    assert "Excel" in labels
    assert "Lean Management" in labels
    assert "PDCA" in labels
    assert "RGPD" in labels
    categories = {item.canonical_label: item.category for item in derived}
    assert categories["Excel"] == "tool"
    assert categories["Lean Management"] == "method"
    assert categories["RGPD"] == "regulatory"


def _build_table_page(page_number: int, activity_text: str, competency_text: str, criteria_text: list[str]) -> ExtractedTablePage:
    columns = {
        "activity": [ExtractedCell("activity", activity_text, page_number)],
        "competency": [ExtractedCell("competency", competency_text, page_number)],
        "criteria": [ExtractedCell("criteria", text, page_number) for text in criteria_text],
    }
    return ExtractedTablePage(page_number=page_number, columns=columns, header_detected=True, layout_quality=0.9)


def test_import_service_infers_title_from_pdf_text(tmp_path, monkeypatch):
    from referential_import import import_service as import_module

    pdf_path = tmp_path / "tmp1nsno2if.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 title infer")

    fake_document = PdfDocument(
        path=pdf_path,
        extraction_method="pdftotext-layout",
        pages=[
            PdfPage(
                number=1,
                text=(
                    "Manager d'affaires REFERENTIEL D'ACTIVITES décrit les situations de travail et les activités exercées, les métiers ou emplois visés\n"
                    "C1.1 Organiser la coordination"
                ),
                blocks=[],
            )
        ],
    )
    fake_tables = [
        ExtractedTablePage(page_number=1, columns={"full_text": [ExtractedCell("full_text", fake_document.pages[0].text, 1)]}, header_detected=False, layout_quality=0.2),
    ]

    monkeypatch.setattr(import_module, "load_pdf_document", lambda path: fake_document)
    monkeypatch.setattr(import_module, "detect_tables", lambda document: fake_tables)

    service = ReferentialImportService(store=ReferentialImportStore(tmp_path / "imports.sqlite3"), output_dir=tmp_path / "out")
    analysis = service.analyze(pdf_path)

    assert analysis["document"].title == "Manager d'affaires"
    assert "tmp1nsno2if" not in analysis["document"].title


def test_import_service_text_fallback_without_columns(tmp_path, monkeypatch):
    from referential_import import import_service as import_module

    pdf_path = tmp_path / "referentiel_fallback.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fallback referential")

    fake_document = PdfDocument(
        path=pdf_path,
        extraction_method="pdftotext-layout",
        pages=[
            PdfPage(
                number=1,
                text=(
                    "Bloc 1\n"
                    "Activité 1\n"
                    "A1.1\n"
                    "C1.1 Organiser la coordination et la coopération des différents intervenants internes et externes en utilisant la méthode du Lean Management pour améliorer la performance de l'entreprise\n"
                    "CE1.1.1 La coordination est structurée\n"
                    "CE1.1.2 La performance est améliorée"
                ),
                blocks=[],
            ),
            PdfPage(
                number=2,
                text=(
                    "C1.2 Déployer Excel pour suivre le pilotage des actions\n"
                    "CE1.2.1 L'outil Excel est mobilisé"
                ),
                blocks=[],
            ),
        ],
    )
    fake_tables = [
        ExtractedTablePage(page_number=1, columns={"full_text": [ExtractedCell("full_text", fake_document.pages[0].text, 1)]}, header_detected=False, layout_quality=0.2),
        ExtractedTablePage(page_number=2, columns={"full_text": [ExtractedCell("full_text", fake_document.pages[1].text, 2)]}, header_detected=False, layout_quality=0.2),
    ]

    monkeypatch.setattr(import_module, "load_pdf_document", lambda path: fake_document)
    monkeypatch.setattr(import_module, "detect_tables", lambda document: fake_tables)

    service = ReferentialImportService(store=ReferentialImportStore(tmp_path / "imports.sqlite3"), output_dir=tmp_path / "out")
    analysis = service.analyze(pdf_path)

    assert analysis["report"].competencies == 2
    assert any(item.code == "C1.1" for item in analysis["competencies"])
    assert any(item.canonical_label == "Excel" for item in analysis["derived_skills"])


def test_import_service_synthetic_analysis_and_dedup(tmp_path, monkeypatch):
    from referential_import import import_service as import_module

    pdf_path = tmp_path / "referentiel.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 synthetic referential")

    fake_document = PdfDocument(
        path=pdf_path,
        extraction_method="pdftotext-bbox-layout",
        pages=[
            PdfPage(
                number=1,
                text=(
                    "Bloc 1\n"
                    "Activité 1\n"
                    "A1.1\n"
                    "C1.1 Organiser la coordination et la coopération des différents intervenants internes et externes en utilisant la méthode du Lean Management pour améliorer la performance de l'entreprise\n"
                    "CE1.1.1 La coordination est structurée\n"
                    "CE1.1.2 La performance est améliorée"
                ),
                blocks=[PdfTextBlock(text="Bloc 1", page_number=1), PdfTextBlock(text="Activité 1", page_number=1), PdfTextBlock(text="A1.1", page_number=1)],
            ),
            PdfPage(
                number=2,
                text=(
                    "C1.2 Déployer Excel pour suivre le pilotage des actions\n"
                    "CE1.2.1 L'outil Excel est mobilisé"
                ),
                blocks=[PdfTextBlock(text="C1.2 Déployer Excel pour suivre le pilotage des actions", page_number=2)],
            ),
        ],
    )
    fake_tables = [
        _build_table_page(
            1,
            "Bloc 1",
            "C1.1 Organiser la coordination et la coopération des différents intervenants internes et externes en utilisant la méthode du Lean Management pour améliorer la performance de l'entreprise",
            ["CE1.1.1 La coordination est structurée", "CE1.1.2 La performance est améliorée"],
        ),
        _build_table_page(
            2,
            "Activité 1\nA1.1",
            "C1.2 Déployer Excel pour suivre le pilotage des actions",
            ["CE1.2.1 L'outil Excel est mobilisé"],
        ),
    ]

    monkeypatch.setattr(import_module, "load_pdf_document", lambda path: fake_document)
    monkeypatch.setattr(import_module, "detect_tables", lambda document: fake_tables)

    store = ReferentialImportStore(tmp_path / "imports.sqlite3")
    service = ReferentialImportService(store=store, output_dir=tmp_path / "out")
    analysis = service.analyze(pdf_path)

    assert analysis["report"].blocks == 1
    assert analysis["report"].activities >= 1
    assert analysis["report"].competencies == 2
    assert analysis["report"].criteria == 3
    assert any(skill.canonical_label == "Lean Management" for skill in analysis["derived_skills"])
    assert any(skill.canonical_label == "Excel" for skill in analysis["derived_skills"])

    output_path = service.approve(analysis, validated_by="tester")
    assert output_path.exists()
    assert store.has_document(analysis["document"].sha256, "0.1.0") is True

    second = service.analyze(pdf_path)
    assert second["duplicate_document"] is True


def test_store_persists_json_payload(tmp_path):
    store = ReferentialImportStore(tmp_path / "imports.sqlite3")
    assert store.list_imports() == []


def test_cli_help_runs():
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo_root / "scripts" / "import_referential.py"), "--help"], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "Importeur spécialisé" in result.stdout


def test_dst_mle_referential_pdf_regression(tmp_path):
    from referential_learning.pdf_loader import load_pdf_document

    pdf_path = Path('data/raw/referentiel/03_DataScientest_Machine_Learning_Engineer.pdf')
    assert pdf_path.exists()

    document = load_pdf_document(pdf_path)
    assert len(document.pages) == 2
    assert sum(len(page.blocks) for page in document.pages) > 30
    assert sum(len(page.text) for page in document.pages) > 1000

    service = ReferentialImportService(store=ReferentialImportStore(tmp_path / 'imports.sqlite3'), output_dir=tmp_path / 'out')
    analysis = service.analyze(pdf_path)
    report = analysis['report']
    metadata = analysis['document']

    assert metadata.provider == 'DataScientest'
    assert metadata.title == 'Machine Learning Engineer'
    assert metadata.reference == 'DST-MLE'
    assert metadata.duration_hours == 560
    assert metadata.cpf_eligible is True
    assert report.status == 'review_required'

    skill_labels = {clean_label for clean_label in [item.canonical_label for item in analysis['derived_skills']]}
    expected = {'Python', 'PyTorch', 'Docker', 'Kubernetes', 'MLflow', 'Cloud', 'MLOps'}
    assert expected.issubset(skill_labels)
    assert not any('Compilation à fins de recherche' in item.label for item in analysis['derived_skills'])


def test_add_skill_with_empty_competencies():
    from referential_import.editing_service import ReferentialEditingService
    from referential_import.models import OfficialCompetency
    
    service = ReferentialEditingService()
    
    analysis = {
        "competencies": [],
        "derived_skills": [],
        "criteria": [],
    }
    
    result = service.add_skill(analysis, "Nouvelle compétence")
    
    assert len(result["competencies"]) == 1
    assert result["competencies"][0].code == "MANUAL_1"
    assert result["competencies"][0].official_label == "Nouvelle compétence"
    assert result["competencies"][0].block_code == "MANUAL"
    assert result["competencies"][0].activity_code == "MANUAL"
    assert result["competencies"][0].page_start == 1
    assert result["competencies"][0].page_end == 1
    assert result["competencies"][0].review_status == "approved"
    assert result["competencies"][0].provenance == "human_review"


def test_add_skill_with_existing_competencies():
    from referential_import.editing_service import ReferentialEditingService
    from referential_import.models import OfficialCompetency
    
    service = ReferentialEditingService()
    
    existing = OfficialCompetency(
        code="C1.1",
        official_label="Compétence existante",
        normalized_label="competence existante",
        block_code="BLOC_1",
        activity_code="A1.1",
        page_start=5,
        page_end=5,
        confidence=0.95,
        source_pages=[5],
    )
    
    analysis = {
        "competencies": [existing],
        "derived_skills": [],
        "criteria": [],
    }
    
    result = service.add_skill(analysis, "Nouvelle compétence")
    
    assert len(result["competencies"]) == 2
    assert result["competencies"][0].code == "C1.1"
    assert result["competencies"][1].code == "MANUAL_1"
    assert result["competencies"][1].official_label == "Nouvelle compétence"
    assert result["competencies"][1].block_code == "BLOC_1"
    assert result["competencies"][1].activity_code == "A1.1"
    assert result["competencies"][1].page_start == 5
    assert result["competencies"][1].page_end == 5
