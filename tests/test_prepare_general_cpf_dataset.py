"""Tests pour la preparation du dataset generaliste CPF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_general_cpf_dataset import (
    build_formation_id,
    build_group_id,
    normalize_text,
    parse_pipe_list,
)


def test_normalize_text():
    assert normalize_text("  Hello   World  ") == "Hello World"
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("Single") == "Single"


def test_parse_pipe_list():
    assert parse_pipe_list("A | B | C") == ["A", "B", "C"]
    assert parse_pipe_list("") == []
    assert parse_pipe_list(None) == []
    assert parse_pipe_list("A|B") == ["A", "B"]


def test_parse_pipe_list_whitespace():
    result = parse_pipe_list("  Python  |  Java  |  ")
    assert result == ["Python", "Java"]


def test_build_formation_id_with_code():
    import pandas as pd
    row = pd.Series({"#": "42", "Code certification": "RNCP123"})
    fid = build_formation_id(row)
    assert "RNCP123" in fid
    assert "42" in fid


def test_build_formation_id_fallback():
    import pandas as pd
    row = pd.Series({"#": "99", "Code certification": ""})
    fid = build_formation_id(row)
    assert fid == "CPF-99"


def test_build_group_id_with_cert_code():
    import pandas as pd
    row = pd.Series({
        "Code certification": "RNCP123",
        "Organisme de formation": "Org",
        "Intitule de la formation": "Title",
    })
    gid = build_group_id(row)
    assert gid == "cert:RNCP123"


def test_build_group_id_with_org_title():
    import pandas as pd
    row = pd.Series({
        "Code certification": "",
        "Organisme de formation": "FormaPlus",
        "Intitule de la formation": "BTS Communication",
    })
    gid = build_group_id(row)
    assert gid.startswith("org:FormaPlus|bts communication")


def test_full_prepare_pipeline(tmp_path):
    """Test minimal de la fonction prepare_dataset avec un petit fichier."""
    import pandas as pd
    from scripts.prepare_general_cpf_dataset import prepare_dataset

    # Create a small test Excel file
    data = {
        "#": [1, 2, 3, 4],
        "Secteur": ["Tech", "Tech", "Health", ""],
        "Organisme de formation": ["OrgA", "OrgA", "OrgB", ""],
        "Intitule de la formation": [
            "Python Avance", "Python Avance", "Data Science", "",
        ],
        "Type de certification": ["RNCP", "RNCP", "", ""],
        "Code certification": ["CERT01", "CERT01", "", ""],
        "Niveau": ["6", "6", "", ""],
        "Codes ROME": ["M1805", "M1805", "M1806", ""],
        "texte_source_competences": [
            "programmation python, machine learning",
            "python approfondi, data science", "analyse de donnees",
            "",
        ],
        "competences_structurees": [
            "Python | Machine Learning",
            "Python | Deep Learning",
            "Python | Data Science",
            "",
        ],
        "Modalite": ["Presentiel", "Presentiel", "Distanciel", ""],
        "Duree": ["1 an", "1 an", "6 mois", ""],
        "Prix TTC (€)": [3000, 3000, 2500, None],
        "Tags": ["python | ml", "python | ml", "data | science", ""],
        "Relue / Valide (oui/non)": ["oui", "oui", "oui", ""],
        "Corrections / Remarques": ["", "", "", ""],
    }
    df = pd.DataFrame(data)
    xlsx_path = tmp_path / "test_dataset.xlsx"
    df.to_excel(xlsx_path, sheet_name="Dataset_Generaliste", index=False)

    # Mock args
    class Args:
        input = str(xlsx_path)
        output_dir = str(tmp_path / "output")
        sheet = "Dataset_Generaliste"

    report = prepare_dataset(Args())

    assert report["total_raw"] == 4
    # Row 4 has no title -> removed
    assert report["total_after_cleaning"] >= 2
    assert "parquet" in report["output_files"]
    assert "jsonl" in report["output_files"]


def test_output_files_exist(tmp_path):
    import pandas as pd
    from scripts.prepare_general_cpf_dataset import prepare_dataset

    data = {
        "#": [1],
        "Secteur": ["Tech"],
        "Organisme de formation": ["OrgA"],
        "Intitule de la formation": ["Formation Test"],
        "Type de certification": ["RNCP"],
        "Code certification": ["CERT01"],
        "Niveau": ["6"],
        "Codes ROME": ["M1805"],
        "texte_source_competences": ["programmation python"],
        "competences_structurees": ["Python"],
        "Modalite": ["Presentiel"],
        "Duree": ["1 an"],
        "Prix TTC (€)": [2000],
        "Tags": ["python"],
    }
    df = pd.DataFrame(data)
    xlsx_path = tmp_path / "test_single.xlsx"
    df.to_excel(xlsx_path, sheet_name="Dataset_Generaliste", index=False)

    class Args:
        input = str(xlsx_path)
        output_dir = str(tmp_path / "output")
        sheet = "Dataset_Generaliste"

    prepare_dataset(Args())

    assert (tmp_path / "output" / "formations_generalistes.parquet").exists()
    assert (tmp_path / "output" / "formations_generalistes.jsonl").exists()
    assert (Path("reports") / "cpf_generaliste_quality_report.json").exists()
