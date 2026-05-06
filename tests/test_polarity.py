"""Tests for the polarity layer.

Most tests use ``patch_polarity`` to swap in the FakePolarityClassifier so
they run without onnxruntime/transformers installed. The PolarityClassifier
itself is import-tested only — actually loading the ONNX model is an
``integration`` test that's skipped by default.
"""

import pytest


def test_polarity_module_imports():
    from keyneg import polarity
    assert hasattr(polarity, "PolarityClassifier")
    assert hasattr(polarity, "PolarityError")
    assert hasattr(polarity, "get_polarity_classifier")


def test_polarity_filter_applied_marker(kn, patch_polarity, negative_doc):
    result = kn.analyze(
        negative_doc,
        polarity_filter=True,
        keyword_threshold=0.0,
        sentiment_threshold=0.0,
    )
    assert result["polarity_filter_applied"] is True
    assert result["polarity_score"] != 0.0  # something was scored


def test_polarity_filter_drops_positive_doc(kn, patch_polarity, positive_doc):
    """The 'great session about preventing burnout' case: a positive doc
    that topically matches negative keywords should produce empty taxonomy
    output once the polarity filter is on.
    """
    result = kn.analyze(
        positive_doc,
        polarity_filter=True,
        polarity_threshold=0.0,
    )
    # Polarity ran; filter dropped the positive sentences.
    assert result["polarity_filter_applied"] is True
    assert result["polarity_score"] > 0  # net positive
    assert result["sentiments"] == []
    assert result["keywords"] == []
    assert result["topic_match_score"] == 0.0


def test_polarity_filter_keeps_negative_sentences(kn, patch_polarity, negative_doc):
    result = kn.analyze(
        negative_doc,
        polarity_filter=True,
        polarity_threshold=0.0,
    )
    assert len(result["negative_sentences"]) > 0
    # All kept sentences should be net-negative under the fake classifier.
    for s in result["negative_sentences"]:
        score = patch_polarity.polarity_score(s)
        assert score < 0


def test_polarity_filter_threshold_strictness(kn, patch_polarity):
    """A stricter (more negative) threshold keeps fewer sentences."""
    doc = (
        "I love this team. "
        "But the toxic culture and the constant micromanagement burn me out."
    )
    loose = kn.analyze(doc, polarity_filter=True, polarity_threshold=0.0)
    strict = kn.analyze(doc, polarity_filter=True, polarity_threshold=-0.5)

    assert len(strict["negative_sentences"]) <= len(loose["negative_sentences"])


def test_polarity_filter_empty_doc(kn, patch_polarity):
    result = kn.analyze("", polarity_filter=True)
    assert result["polarity_filter_applied"] is False
    assert result["polarity_score"] == 0.0
    assert result["negative_sentences"] == []


def test_analyze_batch_with_polarity_filter(kn, patch_polarity, negative_doc, positive_doc):
    results = kn.analyze_batch(
        [negative_doc, positive_doc],
        polarity_filter=True,
        show_progress=False,
    )
    assert len(results) == 2
    assert all(r["polarity_filter_applied"] for r in results)
    # Negative doc should have content; positive doc should be empty.
    assert results[0]["sentiments"] != [] or results[0]["keywords"] != []
    assert results[1]["sentiments"] == []


def test_polarity_classifier_raises_clean_error_without_extras(monkeypatch):
    """If onnxruntime/tokenizers aren't installed, PolarityClassifier()
    should raise PolarityError with a clear install hint.
    """
    from keyneg.polarity import PolarityClassifier, PolarityError

    # Force the ImportError path by hiding onnxruntime.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "onnxruntime":
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(PolarityError) as exc_info:
        PolarityClassifier()
    assert "polarity" in str(exc_info.value).lower()


@pytest.mark.integration
def test_real_polarity_model_loads():
    """Actually load the real ONNX model — slow, requires extras + network.
    Run with ``pytest -m integration``.
    """
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")

    from keyneg.polarity import PolarityClassifier

    clf = PolarityClassifier()
    result = clf.classify("This product is absolutely terrible.")
    assert result["label"] == "NEGATIVE"
    assert result["score"] > 0.5
