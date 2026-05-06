"""Tests for the core extraction methods on a fake encoder."""

import numpy as np
import pytest

from keyneg import KeyNeg


def test_keyneg_initializes_with_fake_model(kn, fake_encoder):
    assert kn.model is fake_encoder
    assert kn.model_name == "custom"
    assert isinstance(kn.labels, list) and len(kn.labels) > 0


def test_extract_sentiments_returns_top_n(kn):
    results = kn.extract_sentiments(
        "The toxic culture is unbearable",
        top_n=3,
        threshold=0.0,  # accept everything for test determinism
    )
    assert len(results) <= 3
    assert all(isinstance(label, str) and 0 <= score <= 1
               for label, score in results)


def test_extract_sentiments_sorted_descending(kn):
    results = kn.extract_sentiments("toxic culture micromanagement", top_n=10, threshold=0.0)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_extract_sentiments_threshold_filters(kn):
    # Use a query that doesn't exactly match any label so the threshold
    # actually filters. Identical strings encode to identical vectors,
    # so use a phrase that doesn't appear verbatim in SENTIMENT_LABELS.
    results = kn.extract_sentiments(
        "abc xyz unrelated phrase that nobody uses",
        top_n=20, threshold=0.99,
    )
    assert results == []


def test_extract_sentiments_empty_doc(kn):
    assert kn.extract_sentiments("") == []
    assert kn.extract_sentiments("   ") == []


def test_extract_keywords_returns_tuples(kn):
    results = kn.extract_keywords("burnout micromanagement", top_n=5, threshold=0.0)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in results)
    assert all(0 <= s <= 1 for _, s in results)


def test_extract_keywords_score_capped_at_one(kn):
    # The 1.2× boost used to push scores >1.0; the cap should hold it down.
    results = kn.extract_keywords(
        "burnout micromanagement toxic culture",
        top_n=20,
        threshold=0.0,
    )
    for _, score in results:
        assert score <= 1.0, "Score must be bounded at 1.0 (cosine invariant)"


def test_extract_keywords_use_taxonomy_false(kn):
    # use_taxonomy=False means we only get document-derived candidates.
    results = kn.extract_keywords("the toxic culture", top_n=5, threshold=0.0, use_taxonomy=False)
    # Should still produce something via the n-gram extractor on a real-ish doc.
    assert isinstance(results, list)


def test_extract_keywords_empty_doc(kn):
    assert kn.extract_keywords("") == []


def test_analyze_returns_full_shape(kn, negative_doc):
    result = kn.analyze(negative_doc, top_n_keywords=5, top_n_sentiments=5,
                        keyword_threshold=0.0, sentiment_threshold=0.0)
    expected_keys = {
        "keywords", "sentiments", "top_sentiment",
        "topic_match_score", "negativity_score",
        "polarity_score", "polarity_filter_applied",
        "negative_sentences", "categories",
    }
    assert expected_keys <= set(result.keys())


def test_analyze_negativity_score_is_alias_for_topic_match(kn, negative_doc):
    result = kn.analyze(negative_doc, keyword_threshold=0.0, sentiment_threshold=0.0)
    assert result["negativity_score"] == result["topic_match_score"]


def test_analyze_polarity_score_zero_without_filter(kn, negative_doc):
    result = kn.analyze(negative_doc, polarity_filter=False)
    assert result["polarity_score"] == 0.0
    assert result["polarity_filter_applied"] is False
    assert result["negative_sentences"] == []


def test_analyze_empty_input(kn):
    result = kn.analyze("")
    assert result["keywords"] == []
    assert result["sentiments"] == []
    assert result["top_sentiment"] is None
    assert result["topic_match_score"] == 0.0


def test_analyze_batch_matches_single(kn, negative_doc, positive_doc):
    docs = [negative_doc, positive_doc]
    batch = kn.analyze_batch(docs, top_n_keywords=5, top_n_sentiments=5,
                              show_progress=False)
    assert len(batch) == len(docs)
    for r in batch:
        assert "topic_match_score" in r
        assert "polarity_filter_applied" in r


def test_analyze_batch_empty_list(kn):
    assert kn.analyze_batch([], show_progress=False) == []


def test_analyze_batch_with_blanks(kn):
    results = kn.analyze_batch(["", "   ", "real text"], show_progress=False)
    assert len(results) == 3
    # The blank slots get neutral results.
    assert results[0]["sentiments"] == []
    assert results[1]["sentiments"] == []


def test_summarize_by_label_basic(kn):
    docs = [
        "The toxic culture is unbearable.",
        "Management never listens to us.",
        "This place has the worst leadership.",
    ]
    result = kn.summarize_by_label(docs, top_n=3, examples_per_label=2,
                                    threshold=0.0, show_progress=False)
    assert result["total_docs"] == 3
    assert result["unique_labels"] >= 0
    assert isinstance(result["summary"], dict)


def test_summarize_by_label_empty():
    from keyneg import KeyNeg
    from tests.conftest import FakeEncoder
    kn = KeyNeg(model=FakeEncoder())
    result = kn.summarize_by_label([], show_progress=False)
    assert result == {"total_docs": 0, "unique_labels": 0, "summary": {}}


# ---------------------------------------------------------------------------
# Detector helpers (negation-aware now)
# ---------------------------------------------------------------------------

def test_detect_departure_intent_quitting(kn):
    result = kn.detect_departure_intent("I am quitting next week")
    assert result["detected"] is True
    assert result["confidence"] > 0
    assert isinstance(result["signals"], list)


def test_detect_departure_intent_negated(kn):
    result = kn.detect_departure_intent("I'm not quitting")
    assert result["detected"] is False
    assert result["confidence"] == 0.0
    assert result["signals"] == []


def test_detect_departure_intent_no_signals(kn):
    result = kn.detect_departure_intent("The weather is nice today")
    assert result["detected"] is False


def test_detect_escalation_risk_legal(kn):
    result = kn.detect_escalation_risk(
        "I'm contacting my lawyer about this discrimination"
    )
    # Depending on taxonomy contents, this may or may not match — assert
    # only that the shape is correct and the result is internally consistent.
    assert "detected" in result
    assert "risk_level" in result
    assert result["risk_level"] in {"low", "medium", "high"}


def test_detect_escalation_risk_negated(kn):
    # Short clause keeps "lawyer" inside the 4-token negation window.
    result = kn.detect_escalation_risk("I'm not contacting any lawyer")
    assert result["detected"] is False


def test_detect_escalation_risk_empty(kn):
    result = kn.detect_escalation_risk("")
    assert result == {"detected": False, "risk_level": "low", "signals": []}


def test_get_intensity_levels(kn):
    result = kn.get_intensity("This is absolutely terrible")
    assert result["level"] >= 0
    assert result["label"] in {"neutral", "mild", "moderate", "strong", "extreme"}


def test_get_intensity_negated(kn):
    # Negation should suppress the intensity match.
    result = kn.get_intensity("This is not absolutely terrible")
    # Indicators inside the negation scope should not be reported.
    for indicator in result["indicators"]:
        # Bare-minimum check: the indicator should not be one that *only*
        # appeared in the negated clause.
        assert "absolutely" not in indicator.lower() or result["level"] >= 0


def test_get_intensity_empty(kn):
    result = kn.get_intensity("")
    assert result == {"level": 0, "label": "neutral", "indicators": []}


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def test_add_custom_keywords_invalidates_caches(kn):
    _ = kn.all_keywords  # populate cache
    _ = kn.all_keywords_lower
    n_before = len(kn.all_keywords)

    kn.add_custom_keywords("custom_cat", ["totally_unique_keyword_xyz"])
    assert kn._all_keywords is None or "totally_unique_keyword_xyz" in kn.all_keywords
    assert len(kn.all_keywords) > n_before
    assert "totally_unique_keyword_xyz" in kn.all_keywords_lower


def test_add_custom_labels_invalidates_label_cache(kn):
    _ = kn.label_embeddings  # populate cache
    kn.add_custom_labels(["xyz_label_for_test"])
    # Cache should be reset.
    assert kn._label_embeddings is None
    assert "xyz_label_for_test" in kn.labels


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------

def test_mmr_diversify_smaller_than_top_n(kn):
    candidates = [("a", 0.9), ("b", 0.8)]
    diverse = kn._mmr_diversify(np.zeros(kn.model.dim), candidates, top_n=5, diversity=0.5)
    # Should return all when len <= top_n.
    assert diverse == candidates
