from __future__ import annotations

from referential_learning.ml_dl_taxonomy import build_taxonomy, canonicalize_term, find_mentions, infer_families


def test_taxonomy_contains_expected_families():
    taxonomy = build_taxonomy()
    labels = [family['label'] for family in taxonomy['families']]
    assert labels == ['Machine Learning', 'Deep Learning', 'NLP', 'MLOps', 'Other']


def test_alias_detection_on_title():
    text = 'Data Scientist - Machine Learning & Deep Learning'
    mentions = find_mentions(text)
    families = infer_families(text)
    assert any(item['canonical_name'] == 'Machine Learning' for item in mentions)
    assert any(item['canonical_name'] == 'Deep Learning' for item in mentions)
    assert 'Machine Learning' in families
    assert 'Deep Learning' in families


def test_canonicalize_aliases():
    assert canonicalize_term('Tensor Flow') == ('TensorFlow', 'Deep Learning', 'TOOL')
    assert canonicalize_term('apprentissage automatique') == ('Machine Learning', 'Machine Learning', 'DOMAIN')
