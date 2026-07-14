from __future__ import annotations

from referential_import.pdf_loader import PdfDocument, PdfPage, PdfTextBlock
from referential_import.title_extractor import extract_referential_title


def _doc(tmp_path, name: str, pages: list[PdfPage]) -> PdfDocument:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 fake")
    return PdfDocument(path=path, pages=pages, extraction_method="pdftotext-bbox-layout")


def test_explicit_certification_label_wins_over_generic_header(tmp_path):
    document = _doc(
        tmp_path,
        "referentiel_certification.pdf",
        [
            PdfPage(
                number=1,
                width=595,
                height=842,
                text=(
                    "RÉFÉRENTIEL DE CERTIFICATION\n"
                    "INTITULÉ DE LA CERTIFICATION\n"
                    "Ingénieur en intelligence artificielle"
                ),
                blocks=[
                    PdfTextBlock(text="RÉFÉRENTIEL DE CERTIFICATION", page_number=1, order=0, font_size=20, bold=True, bbox=(120, 760, 480, 790)),
                    PdfTextBlock(text="INTITULÉ DE LA CERTIFICATION", page_number=1, order=1, font_size=18, bold=True, bbox=(110, 720, 500, 748)),
                    PdfTextBlock(text="Ingénieur en intelligence artificielle", page_number=1, order=2, font_size=14, bold=True, bbox=(120, 680, 480, 706)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="referentiel_certification.pdf")
    assert result.title == "Ingénieur en intelligence artificielle"
    assert result.title_type == "certification_title"
    assert result.confidence >= 0.8
    assert result.source_page == 1
    assert any("explicit_label" in reason for reason in result.candidates[0].reasons)


def test_colon_form_is_extracted(tmp_path):
    document = _doc(
        tmp_path,
        "manager_dev_commercial.pdf",
        [
            PdfPage(
                number=1,
                text="INTITULÉ DE LA CERTIFICATION : Manager du développement commercial",
                blocks=[PdfTextBlock(text="INTITULÉ DE LA CERTIFICATION : Manager du développement commercial", page_number=1, order=0, font_size=16, bold=True, bbox=(100, 720, 500, 750))],
            )
        ],
    )
    result = extract_referential_title(document, file_name="manager_dev_commercial.pdf")
    assert result.title == "Manager du développement commercial"
    assert result.title_type == "certification_title"
    assert result.confidence >= 0.8


def test_rncp_context_prefers_title_over_generic_fiche(tmp_path):
    document = _doc(
        tmp_path,
        "fiche_rncp.pdf",
        [
            PdfPage(
                number=1,
                text=(
                    "FICHE RNCP\n"
                    "RNCP41966\n"
                    "Manager du développement commercial"
                ),
                blocks=[
                    PdfTextBlock(text="FICHE RNCP", page_number=1, order=0, font_size=18, bold=True, bbox=(120, 760, 460, 790)),
                    PdfTextBlock(text="RNCP41966", page_number=1, order=1, font_size=14, bold=True, bbox=(120, 730, 260, 752)),
                    PdfTextBlock(text="Manager du développement commercial", page_number=1, order=2, font_size=16, bold=True, bbox=(120, 680, 500, 708)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="fiche_rncp.pdf")
    assert result.title == "Manager du développement commercial"
    assert result.rncp_code == "RNCP41966"
    assert result.confidence >= 0.7


def test_block_title_is_not_selected_over_fallback(tmp_path):
    document = _doc(
        tmp_path,
        "dossier_referentiel.pdf",
        [
            PdfPage(
                number=1,
                text="Référentiel de compétences\nBloc 1 - Piloter une stratégie commerciale",
                blocks=[
                    PdfTextBlock(text="Référentiel de compétences", page_number=1, order=0, font_size=20, bold=True, bbox=(120, 760, 420, 790)),
                    PdfTextBlock(text="Bloc 1 - Piloter une stratégie commerciale", page_number=1, order=1, font_size=18, bold=True, bbox=(120, 700, 500, 730)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="dossier_referentiel.pdf")
    assert result.title != "Bloc 1 - Piloter une stratégie commerciale"
    assert result.confidence < 0.7
    assert any("Aucun libellé explicite trouvé." in warning for warning in result.warnings)


def test_multiline_title_is_concatenated(tmp_path):
    document = _doc(
        tmp_path,
        "chef_projet_ia.pdf",
        [
            PdfPage(
                number=1,
                text=(
                    "Nom de la certification\n"
                    "Chef de projet\n"
                    "en intelligence artificielle"
                ),
                blocks=[
                    PdfTextBlock(text="Nom de la certification", page_number=1, order=0, font_size=16, bold=True, bbox=(120, 760, 390, 785)),
                    PdfTextBlock(text="Chef de projet", page_number=1, order=1, font_size=14, bold=True, bbox=(120, 720, 290, 745)),
                    PdfTextBlock(text="en intelligence artificielle", page_number=1, order=2, font_size=14, bold=True, bbox=(120, 690, 430, 715)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="chef_projet_ia.pdf")
    assert result.title == "Chef de projet en intelligence artificielle"
    assert result.title_type == "certification_title"


def test_filename_fallback_is_low_confidence(tmp_path):
    document = _doc(
        tmp_path,
        "fichier_sans_titre.pdf",
        [PdfPage(number=1, text="Le document ne contient aucun intitulé identifiable.", blocks=[])]
    )
    result = extract_referential_title(document, file_name="fichier_sans_titre.pdf")
    assert result.title == "fichier sans titre"
    assert result.confidence < 0.5
    assert any("Résultat issu du nom du fichier." in warning for warning in result.warnings)


def test_visual_title_is_not_overridden_by_bigger_certificatee_name(tmp_path):
    document = _doc(
        tmp_path,
        "visuel.pdf",
        [
            PdfPage(
                number=1,
                width=595,
                height=842,
                text=(
                    "CERTIFICATEUR EXTERNE\n"
                    "Intitulé de la certification\n"
                    "Expert en cybersécurité"
                ),
                blocks=[
                    PdfTextBlock(text="CERTIFICATEUR EXTERNE", page_number=1, order=0, font_size=24, bold=True, bbox=(80, 760, 520, 790)),
                    PdfTextBlock(text="Intitulé de la certification", page_number=1, order=1, font_size=18, bold=True, bbox=(90, 700, 470, 724)),
                    PdfTextBlock(text="Expert en cybersécurité", page_number=1, order=2, font_size=14, bold=True, bbox=(120, 660, 430, 686)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="visuel.pdf")
    assert result.title == "Expert en cybersécurité"
    assert result.title_type == "certification_title"


def test_competency_blocks_do_not_replace_main_title(tmp_path):
    document = _doc(
        tmp_path,
        "competencies_only.pdf",
        [
            PdfPage(
                number=1,
                text=(
                    "Référentiel de compétences\n"
                    "Bloc 1 - Piloter une stratégie commerciale\n"
                    "Bloc 2 - Déployer les actions commerciales"
                ),
                blocks=[
                    PdfTextBlock(text="Référentiel de compétences", page_number=1, order=0, font_size=20, bold=True, bbox=(120, 760, 420, 790)),
                    PdfTextBlock(text="Bloc 1 - Piloter une stratégie commerciale", page_number=1, order=1, font_size=18, bold=True, bbox=(120, 700, 500, 730)),
                    PdfTextBlock(text="Bloc 2 - Déployer les actions commerciales", page_number=1, order=2, font_size=18, bold=True, bbox=(120, 660, 500, 690)),
                ],
            )
        ],
    )
    result = extract_referential_title(document, file_name="competencies_only.pdf")
    assert result.title != "Bloc 1 - Piloter une stratégie commerciale"
    assert result.title != "Bloc 2 - Déployer les actions commerciales"
    assert result.confidence < 0.7
