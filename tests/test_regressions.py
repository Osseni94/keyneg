"""Regressions for bugs identified in the v1.1 → v1.2 review.

Each test pins a previously-broken behavior so we don't reintroduce it.
"""

import copy

import pytest

from keyneg import KeyNeg, NEGATIVE_TAXONOMY


def test_score_never_exceeds_one_with_boost(kn):
    """Pre-1.2 the 1.2× boost on taxonomy-matched candidates could push
    cosine scores above 1.0, breaking downstream code that assumed [0,1].
    """
    results = kn.extract_keywords(
        "burnout toxic culture micromanagement harassment discrimination",
        top_n=50,
        threshold=0.0,
    )
    for _, score in results:
        assert score <= 1.0


def test_taxonomy_isolated_between_instances(fake_encoder):
    """Pre-1.2 ``self.taxonomy = NEGATIVE_TAXONOMY.copy()`` was shallow,
    so add_custom_keywords on one KeyNeg leaked into the next.
    """
    snapshot = copy.deepcopy(NEGATIVE_TAXONOMY)

    a = KeyNeg(model=fake_encoder)
    b = KeyNeg(model=fake_encoder)

    a.add_custom_keywords("isolation_test_category", ["leaky_keyword_aaa"])

    # b's taxonomy should NOT see the leaked keyword.
    b_flat = []
    for cat in b.taxonomy.values():
        if isinstance(cat, dict):
            for sub in cat.values():
                if isinstance(sub, list):
                    b_flat.extend(sub)
    assert "leaky_keyword_aaa" not in b_flat

    # The module-level constant should also be unmodified.
    assert NEGATIVE_TAXONOMY == snapshot


def test_lowercase_keyword_set_is_cached(kn):
    """Pre-1.2 the lowercased keyword list was rebuilt on every
    extract_keywords call. Now it's an O(1)-membership cached set.
    """
    first = kn.all_keywords_lower
    second = kn.all_keywords_lower
    assert first is second  # same object, no rebuild
    assert isinstance(first, set)


def test_lowercase_cache_invalidated_after_custom_keywords(kn):
    _ = kn.all_keywords_lower
    kn.add_custom_keywords("regression_test_cat", ["regression_keyword_zzz"])
    assert "regression_keyword_zzz" in kn.all_keywords_lower


def test_not_quitting_does_not_trip_departure_intent(kn):
    """The flagship negation case from the data scientists' critique."""
    assert kn.detect_departure_intent("I'm not quitting")["detected"] is False


def test_no_lawyer_does_not_trip_escalation(kn):
    assert kn.detect_escalation_risk(
        "We are not contacting any lawyer."
    )["detected"] is False


def test_analyze_emits_polarity_fields_even_when_disabled(kn, negative_doc):
    """The new fields must be present on every analyze() result for
    downstream callers that read them unconditionally.
    """
    result = kn.analyze(negative_doc)
    for key in ("topic_match_score", "polarity_score", "polarity_filter_applied",
                "negative_sentences"):
        assert key in result


def test_negativity_score_remains_numeric_alias(kn, negative_doc):
    result = kn.analyze(negative_doc)
    assert isinstance(result["negativity_score"], float)
    assert result["negativity_score"] == result["topic_match_score"]


def test_extract_candidates_returns_list_for_stop_word_only_doc(kn):
    """Pre-1.2 a bare ``except: pass`` swallowed CountVectorizer's
    ValueError on stop-word-only input. We now log it and return [].
    """
    # All English stopwords → CountVectorizer raises ValueError.
    candidates = kn._extract_candidates("the and of to a in", ngram_range=(1, 1))
    assert candidates == []
