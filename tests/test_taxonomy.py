"""Tests pour la taxonomie hiérarchique des compétences IA."""

import json
from pathlib import Path

import pytest

TAXONOMY_PATH = Path("data/referentials/ai_skill_taxonomy.json")


@pytest.fixture
def taxonomy():
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _all_skills(taxonomy):
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            yield skill, family


def test_taxonomy_exists():
    assert TAXONOMY_PATH.exists()
    assert TAXONOMY_PATH.stat().st_size > 100


def test_taxonomy_has_version(taxonomy):
    assert "version" in taxonomy
    assert taxonomy["version"] != ""


def test_taxonomy_version_is_string(taxonomy):
    assert isinstance(taxonomy["version"], str)


def test_taxonomy_has_families(taxonomy):
    assert "families" in taxonomy
    assert len(taxonomy["families"]) >= 5


def test_each_family_has_id(taxonomy):
    for family in taxonomy["families"]:
        assert "id" in family, f"Family missing id: {family.get('label', '?')}"
        assert isinstance(family["id"], str)
        assert len(family["id"]) > 0


def test_each_family_has_label(taxonomy):
    for family in taxonomy["families"]:
        assert "label" in family, f"Family missing label: {family.get('id', '?')}"
        assert isinstance(family["label"], str)


def test_each_family_has_skills(taxonomy):
    for family in taxonomy["families"]:
        assert "skills" in family
        assert len(family["skills"]) >= 1, f"Empty skills in family {family['id']}"


def test_unique_skill_ids(taxonomy):
    ids = [skill["id"] for skill, _ in _all_skills(taxonomy)]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert len(duplicates) == 0, f"Duplicate skill IDs: {duplicates}"


def test_each_skill_has_id(taxonomy):
    for skill, family in _all_skills(taxonomy):
        assert "id" in skill, f"Skill missing id in family {family['id']}"
        assert isinstance(skill["id"], str)


def test_each_skill_has_label(taxonomy):
    for skill, family in _all_skills(taxonomy):
        assert "label" in skill, f"Skill missing label: {skill.get('id', '?')} in {family['id']}"
        assert isinstance(skill["label"], str)


def test_each_skill_has_active_flag(taxonomy):
    for skill, family in _all_skills(taxonomy):
        assert "active" in skill, f"Missing active flag: {skill['id']}"
        assert isinstance(skill["active"], bool)


def test_each_skill_has_aliases(taxonomy):
    for skill, family in _all_skills(taxonomy):
        assert "aliases" in skill, f"Missing aliases: {skill['id']}"
        assert isinstance(skill["aliases"], list)


def test_alias_no_empty_strings(taxonomy):
    for skill, family in _all_skills(taxonomy):
        for alias in skill["aliases"]:
            assert len(alias) > 0, f"Empty alias in {skill['id']}"


def test_no_cycles_in_id_format(taxonomy):
    """Check that no skill ID is a prefix of another skill ID in the same family."""
    for family in taxonomy["families"]:
        ids = [skill["id"] for skill in family["skills"]]
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i != j and (id1.startswith(id2 + ".") or id2.startswith(id1 + ".")):
                    pytest.fail(
                        f"Potential cycle or parent/child confusion: "
                        f"'{id1}' and '{id2}' in family '{family['id']}'"
                    )


def test_source_18_labels_are_valid(taxonomy):
    """Each source_18_label should correspond to a known 18-label name."""
    from scripts.audit_labels import load_dataset, compute_label_frequencies
    from pathlib import Path

    csv_path = Path("data/processed/dataset_entrainement.csv")
    if not csv_path.exists():
        pytest.skip("Dataset non disponible")

    df = load_dataset(csv_path)
    stats = compute_label_frequencies(df)

    valid_18_labels = set(stats.keys())
    
    for skill, _ in _all_skills(taxonomy):
        src = skill.get("source_18_label")
        if src:
            assert src in valid_18_labels, (
                f"source_18_label '{src}' (skill {skill['id']}) "
                f"does not match any known 18-label. Valid: {valid_18_labels}"
            )


def test_activation_levels_consistent(taxonomy):
    """activation_level should be consistent with frequency_dataset."""
    for skill, _ in _all_skills(taxonomy):
        level = skill.get("activation_level")
        freq = skill.get("frequency_dataset", 0)
        if level == "actif":
            assert freq >= 50, f"actif but freq={freq} for {skill['id']}"
        elif level == "experimental":
            assert 10 <= freq < 50, f"experimental but freq={freq} for {skill['id']}"
        elif level == "inactif":
            assert freq < 10, f"inactif but freq={freq} for {skill['id']}"


def test_active_labels_have_sufficient_data(taxonomy):
    """Active labels should have at least 10 examples in the dataset."""
    for skill, _ in _all_skills(taxonomy):
        if skill.get("active", False):
            freq = skill.get("frequency_dataset", 0)
            assert freq >= 10, (
                f"Label actif '{skill['id']}' has only {freq} examples. "
                "Minimum 10 required for experimental, 50 for full activation."
            )


def test_source_18_mapping_complete(taxonomy):
    """Every 18-label should map to a taxonomy entry."""
    from scripts.audit_labels import load_dataset, compute_label_frequencies
    from pathlib import Path

    csv_path = Path("data/processed/dataset_entrainement.csv")
    if not csv_path.exists():
        pytest.skip("Dataset non disponible")

    df = load_dataset(csv_path)
    stats = compute_label_frequencies(df)

    mapped = set()
    for skill, _ in _all_skills(taxonomy):
        src = skill.get("source_18_label")
        if src:
            mapped.add(src)

    for label in stats:
        assert label in mapped, f"18-label '{label}' has no taxonomy mapping"


def test_meta_section_exists(taxonomy):
    assert "meta" in taxonomy
    meta = taxonomy["meta"]
    assert "source_18_labels_mapping" in meta
    assert "activation_criteria" in meta


def test_source_18_labels_mapping_complete(taxonomy):
    meta = taxonomy["meta"]
    mapping = meta.get("source_18_labels_mapping", {})
    assert len(mapping) >= 18, f"Only {len(mapping)} mappings, expected 18"


def test_label_candidates_csv_exists():
    """Optional: label candidates file may not exist yet."""
    path = Path("reports/ai_skill_label_candidates.csv")
    if path.exists():
        assert path.stat().st_size > 0


def test_prepared_dataset_exists():
    path = Path("data/multilabel/multilabel_dataset_info.json")
    if path.exists():
        info = json.loads(path.read_text())
        assert "num_labels" in info
        assert "pos_weight" in info
        assert "id2label" in info
        assert "label2id" in info


def test_multi_hot_consistency():
    """Multi-hot vectors should match label2id order."""
    info_path = Path("data/multilabel/multilabel_dataset_info.json")
    dataset_path = Path("data/multilabel/multilabel_dataset.csv")
    if not info_path.exists() or not dataset_path.exists():
        pytest.skip("Dataset non disponible")

    import pandas as pd
    info = json.loads(info_path.read_text())
    df = pd.read_csv(dataset_path)
    num_labels = info["num_labels"]

    import ast
    first_row = df.iloc[0]["multi_hot"]
    if isinstance(first_row, str):
        first_vector = ast.literal_eval(first_row)
    else:
        first_vector = first_row
    assert len(first_vector) == num_labels, (
        f"Multi-hot vector length {len(first_vector)} != num_labels {num_labels}"
    )


def test_pos_weight_length():
    """pos_weight length should match num_labels."""
    info_path = Path("data/multilabel/multilabel_dataset_info.json")
    if not info_path.exists():
        pytest.skip("Dataset non disponible")
    info = json.loads(info_path.read_text())
    pw = info["pos_weight"]
    assert len(pw) == info["num_labels"]


def test_taxonomy_hash_stable():
    """Same taxonomy = same hash."""
    from scripts.prepare_multilabel_dataset import taxonomy_hash
    import json
    tax = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    h1 = taxonomy_hash(tax)
    h2 = taxonomy_hash(tax)
    assert h1 == h2


def test_audit_report_generated():
    path = Path("reports/label_audit_report.json")
    if path.exists():
        report = json.loads(path.read_text())
        assert "per_label_statistics" in report
        assert "dataset" in report
        assert "summary" in report
