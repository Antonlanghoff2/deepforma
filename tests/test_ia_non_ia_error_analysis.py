from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.ia_non_ia_shared import error_analysis


EXPECTED_COLUMNS = [
    'source_index',
    'record_id',
    'split',
    'text',
    'true_label',
    'predicted_label',
    'probability_ia',
    'threshold',
    'error_type',
]


def test_error_analysis_with_consecutive_index():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'source_index': [0, 1, 2],
            'record_id': ['r0', 'r1', 'r2'],
            'split': ['test', 'test', 'test'],
        },
        index=[0, 1, 2],
    )
    scores = np.array([0.1, 0.9, 0.8])

    result = error_analysis(frame, scores, threshold=0.5)

    assert list(result.columns) == EXPECTED_COLUMNS
    assert len(result) == 1
    assert result.iloc[0]['source_index'] == 2
    assert result.iloc[0]['predicted_label'] == 'IA'
    assert result.iloc[0]['error_type'] == 'faux positif'


def test_error_analysis_with_non_consecutive_index_and_source_index():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'source_index': [12, 312, 900],
            'record_id': ['r12', 'r312', 'r900'],
            'split': ['test', 'test', 'test'],
        },
        index=[12, 312, 900],
    )
    scores = np.array([0.1, 0.4, 0.8])

    result = error_analysis(frame, scores, threshold=0.5)

    assert len(result) == 2
    assert set(result['source_index']) == {312, 900}
    assert set(result['error_type']) == {'faux négatif', 'faux positif'}
    assert result.loc[result['source_index'] == 312, 'probability_ia'].iloc[0] == pytest.approx(0.4)
    assert result.loc[result['source_index'] == 900, 'probability_ia'].iloc[0] == pytest.approx(0.8)


def test_error_analysis_with_non_consecutive_index_without_source_index_column():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'record_id': ['r12', 'r312', 'r900'],
            'split': ['test', 'test', 'test'],
        },
        index=[12, 312, 900],
    )
    scores = [0.1, 0.4, 0.8]

    result = error_analysis(frame, scores, threshold=0.5)

    assert len(result) == 2
    assert list(result['source_index']) == [900, 312]


def test_error_analysis_returns_empty_frame_when_no_errors():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'source_index': [12, 312, 900],
            'record_id': ['r12', 'r312', 'r900'],
            'split': ['test', 'test', 'test'],
        },
        index=[12, 312, 900],
    )
    scores = np.array([0.1, 0.9, 0.2])

    result = error_analysis(frame, scores, threshold=0.5)

    assert result.empty
    assert list(result.columns) == EXPECTED_COLUMNS


@pytest.mark.parametrize(
    'scores, expected_types',
    [
        (np.array([0.9, 0.2, 0.8]), {'faux positif', 'faux négatif'}),
        (np.array([[0.9], [0.2], [0.8]]), {'faux positif', 'faux négatif'}),
    ],
)
def test_error_analysis_handles_multiple_misclassifications(scores, expected_types):
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 1],
            'text': ['A', 'B', 'C'],
            'source_index': [10, 20, 30],
            'record_id': ['r10', 'r20', 'r30'],
            'split': ['test', 'test', 'test'],
        },
        index=[10, 20, 30],
    )

    result = error_analysis(frame, scores, threshold=0.5)

    assert set(result['error_type']) == expected_types
    assert len(result) == 2
    assert result['probability_ia'].tolist() == sorted(result['probability_ia'].tolist(), reverse=True)


def test_error_analysis_raises_on_length_mismatch():
    frame = pd.DataFrame({'is_ai': [0, 1, 0], 'text': ['A', 'B', 'C']}, index=[0, 1, 2])

    with pytest.raises(ValueError, match='Nombre de lignes et de scores incompatibles'):
        error_analysis(frame, np.array([0.1, 0.2]), threshold=0.5)


def test_error_analysis_supports_list_scores():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'source_index': [12, 312, 900],
            'record_id': ['r12', 'r312', 'r900'],
            'split': ['test', 'test', 'test'],
        },
        index=[12, 312, 900],
    )

    result = error_analysis(frame, [0.1, 0.4, 0.8], threshold=0.5)

    assert len(result) == 2
    assert result['probability_ia'].tolist() == [0.8, 0.4]


def test_error_analysis_supports_2d_numpy_scores():
    frame = pd.DataFrame(
        {
            'is_ai': [0, 1, 0],
            'text': ['A', 'B', 'C'],
            'source_index': [12, 312, 900],
            'record_id': ['r12', 'r312', 'r900'],
            'split': ['test', 'test', 'test'],
        },
        index=[12, 312, 900],
    )

    result = error_analysis(frame, np.array([[0.1], [0.4], [0.8]]), threshold=0.5)

    assert len(result) == 2
    assert result['source_index'].tolist() == [900, 312]
