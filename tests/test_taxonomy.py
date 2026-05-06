"""Tests for the taxonomy module."""

import pytest

from keyneg.taxonomy import (
    NEGATIVE_TAXONOMY,
    SENTIMENT_LABELS,
    get_all_keywords,
    get_category_labels,
    get_keywords_by_category,
)


def test_sentiment_labels_nonempty():
    assert isinstance(SENTIMENT_LABELS, list)
    assert len(SENTIMENT_LABELS) > 0
    assert all(isinstance(s, str) and s for s in SENTIMENT_LABELS)


def test_negative_taxonomy_is_dict_of_dicts():
    assert isinstance(NEGATIVE_TAXONOMY, dict)
    assert len(NEGATIVE_TAXONOMY) > 0
    for category, body in NEGATIVE_TAXONOMY.items():
        assert isinstance(category, str)
        assert isinstance(body, dict)


def test_get_all_keywords_is_flat_unique():
    keywords = get_all_keywords()
    assert isinstance(keywords, list)
    assert all(isinstance(k, str) for k in keywords)
    # The function returns a deduplicated set internally; this asserts it.
    assert len(keywords) == len(set(keywords))


def test_get_category_labels_returns_top_level_categories():
    cats = get_category_labels()
    assert isinstance(cats, list)
    for cat in cats:
        assert cat in NEGATIVE_TAXONOMY


def test_get_keywords_by_category_returns_flat_list():
    cats = get_category_labels()
    if not cats:
        pytest.skip("No categories defined")
    sample = cats[0]
    kws = get_keywords_by_category(sample)
    assert isinstance(kws, list)


def test_action_indicators_present_for_detectors():
    """``detect_departure_intent`` and ``detect_escalation_risk`` rely on
    the ``action_indicators`` category being present.
    """
    assert "action_indicators" in NEGATIVE_TAXONOMY
    body = NEGATIVE_TAXONOMY["action_indicators"]
    assert "departure_intent" in body
    assert "escalation_threats" in body
    assert isinstance(body["departure_intent"], list)
    assert len(body["departure_intent"]) > 0


def test_emotional_states_present_for_intensity():
    """``get_intensity`` reads from ``emotional_states.intensity_expressions``."""
    assert "emotional_states" in NEGATIVE_TAXONOMY
    body = NEGATIVE_TAXONOMY["emotional_states"]
    assert "intensity_expressions" in body
    intensity = body["intensity_expressions"]
    assert isinstance(intensity, dict)
    for level in ("mild", "moderate", "strong", "extreme"):
        assert level in intensity
