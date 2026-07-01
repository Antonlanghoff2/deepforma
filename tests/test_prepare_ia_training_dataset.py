"""Tests pour la preparation du dataset d'entrainement IA."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_ia_training_dataset import (
    ALIAS_MAP,
    build_group_id,
    build_text,
    compute_pos_weights,
    load_taxonomy,
    normalize_label,
    normalize_labels,
    verify_no_leak,
)

TAXONOMY_PATH = Path("config/ia_taxonomy_v2.json")


@pytest.fixture
def taxonomy():
    return load_taxonomy(TAXONOMY_PATH)


@pytest.fixture
def label_set(taxonomy):
    return set(taxonomy["labels"])


def test_taxonomy_file_exists():
    assert TAXONOMY_PATH.exists()
    assert TAXONOMY_PATH.stat().st_size > 0


def test_taxonomy_has_20_labels(taxonomy):
    assert len(taxonomy["labels"]) == 20
    assert taxonomy["taxonomy_version"] == "2.0"


def test_taxonomy_no_duplicates(taxonomy):
    assert len(taxonomy["labels"]) == len(set(taxonomy["labels"]))


def test_normalize_label_exact(label_set):
    assert normalize_label("Python", label_set) == "Python"
    assert normalize_label("Machine Learning", label_set) == "Machine Learning"


def test_normalize_label_via_alias(label_set):
    assert normalize_label("NLP", label_set) == "NLP / Traitement du langage"
    assert normalize_label("MLOps", label_set) == "MLOps / Deploiement"
    assert normalize_label("LangChain", label_set) == "LangChain / Agents RAG"


def test_normalize_label_whitespace(label_set):
    assert normalize_label("  Python  ", label_set) == "Python"


def test_normalize_label_unknown(label_set):
    assert normalize_label("Unknown Skill", label_set) is None
    assert normalize_label("", label_set) is None


def test_normalize_labels_pipe_separated(label_set):
    result = normalize_labels("Python | Machine Learning | Unknown", label_set)
    assert result == ["Machine Learning", "Python"]


def test_normalize_labels_empty(label_set):
    assert normalize_labels("", label_set) == []


def test_normalize_labels_nan(label_set):
    import pandas as pd
    assert normalize_labels(pd.NA, label_set) == []


def test_build_text_all_fields():
    import pandas as pd
    row = pd.Series({
        "Intitule de la formation": "Formation Python",
        "Secteur": "Tech",
        "Organisme de formation": "OrgXYZ",
        "Type de certification": "RNCP",
        "Niveau": "6",
        "Codes ROME": "M1805",
        "Tags TrendRadar": "python, data",
    })
    text = build_text(row)
    assert "[TITRE] Formation Python" in text
    assert "[SECTEUR] Tech" in text
    assert "[ORGANISME] OrgXYZ" in text
    assert "[CERTIFICATION] RNCP" in text
    assert "[NIVEAU] 6" in text
    assert "[ROME] M1805" in text
    assert "[TAGS] python, data" in text


def test_build_text_minimal():
    import pandas as pd
    row = pd.Series({
        "Intitule de la formation": "Formation Python",
        "Secteur": "",
        "Organisme de formation": "",
        "Type de certification": "",
        "Niveau": "",
        "Codes ROME": "",
        "Tags TrendRadar": "",
    })
    text = build_text(row)
    assert "[TITRE] Formation Python" in text
    assert "[SECTEUR]" not in text


def test_build_group_id_with_code():
    import pandas as pd
    row = pd.Series({
        "Code certification": "CERT123",
        "Organisme de formation": "Org",
        "#": "42",
    })
    gid = build_group_id(row)
    assert gid == "cert:CERT123"


def test_build_group_id_with_org_title():
    import pandas as pd
    row = pd.Series({
        "Code certification": "",
        "Organisme de formation": "OrgXYZ",
        "Intitule de la formation": "Formation Python",
        "#": "42",
    })
    gid = build_group_id(row)
    assert gid.startswith("org:OrgXYZ|formation python")


def test_build_group_id_fallback():
    import pandas as pd
    row = pd.Series({
        "Code certification": "",
        "Organisme de formation": "",
        "Intitule de la formation": "",
        "#": "99",
    })
    gid = build_group_id(row)
    assert gid == "idx:99"


def test_verify_no_leak_passes():
    import pandas as pd
    splits = {
        "train": pd.DataFrame({"_group_id": ["a", "b", "c"]}),
        "val": pd.DataFrame({"_group_id": ["d", "e"]}),
        "test": pd.DataFrame({"_group_id": ["f", "g"]}),
    }
    verify_no_leak(splits)  # should not raise


def test_verify_no_leak_raises():
    import pandas as pd
    splits = {
        "train": pd.DataFrame({"_group_id": ["a", "b", "c"]}),
        "test": pd.DataFrame({"_group_id": ["c", "d"]}),
    }
    with pytest.raises(ValueError, match="FUITE"):
        verify_no_leak(splits)


def test_compute_pos_weights():
    import numpy as np
    y = np.array([
        [1, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float32)
    weights = compute_pos_weights(y, cap=10.0)
    assert len(weights) == 3
    assert weights[0] == pytest.approx(1.0)  # 2 pos, 2 neg
    assert weights[1] == 3.0  # 1 pos, 3 neg -> 3.0
    assert weights[2] == pytest.approx(1.0)  # 2 pos, 2 neg


def test_compute_pos_weights_cap():
    import numpy as np
    y = np.array([
        [1, 0],
        [0, 0],
        [0, 0],
    ], dtype=np.float32)  # 1 pos, 2 neg -> weight = 2.0
    weights = compute_pos_weights(y, cap=1.5)
    assert weights[0] == 1.5


def test_alias_map_covers_all_labels(taxonomy):
    """Chaque label de la taxonomie doit avoir au moins un alias ou etre dans label_set."""
    label_set = set(taxonomy["labels"])
    for lbl in taxonomy["labels"]:
        key = lbl.lower().strip()
        assert key in ALIAS_MAP or lbl in label_set, f"Label sans alias: {lbl}"
