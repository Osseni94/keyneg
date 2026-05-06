"""Shared test fixtures.

The KeyNeg constructor accepts any object with a ``.encode(texts,
show_progress_bar=...)`` method, which lets us swap in a deterministic
fake encoder for tests instead of paying the cost of loading a real
sentence-transformer model.
"""

import hashlib
from typing import Iterable, List, Union

import numpy as np
import pytest


class FakeEncoder:
    """Deterministic, hash-based stand-in for SentenceTransformer.

    Produces L2-normalized ``dim``-d vectors keyed off ``md5(text)``. Two
    *identical* strings encode to the same vector (cosine 1.0); different
    strings encode to near-orthogonal vectors. This is sufficient for
    exercising the plumbing of similarity-based extraction without testing
    semantic accuracy (which would require a real model).
    """

    def __init__(self, dim: int = 32, seed: int = 1729):
        self.dim = dim
        self.max_seq_length = 512
        self._seed = seed

    def encode(self,
               texts: Union[str, Iterable[str]],
               show_progress_bar: bool = False,
               **_kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.md5(f"{self._seed}:{t}".encode("utf-8")).digest()
            # Expand digest to dim by repeating — md5 is 16 bytes.
            buf = (digest * ((self.dim // 16) + 1))[: self.dim]
            vec = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            vec = (vec - vec.mean()) / 128.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out[i] = vec
        return out

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def kn(fake_encoder):
    """KeyNeg instance backed by the fake encoder."""
    from keyneg import KeyNeg
    return KeyNeg(model=fake_encoder)


@pytest.fixture
def negative_doc() -> str:
    return (
        "I'm completely burned out from the constant micromanagement. "
        "My manager never listens to feedback and takes credit for my work. "
        "The toxic culture here is unbearable and I'm seriously considering quitting."
    )


@pytest.fixture
def positive_doc() -> str:
    # Pure-positive sentences — kept short and without any negative-cue
    # words at all so the FakePolarityClassifier classifies them all
    # positive without ambiguity.
    return (
        "Today was wonderful. "
        "Leadership is supportive and excellent. "
        "The team is happy and thriving."
    )


@pytest.fixture
def empty_docs() -> List[str]:
    return ["", "   ", "\n\t  "]


class FakePolarityClassifier:
    """Deterministic polarity stand-in: keyword-based scoring.

    Returns negative for sentences containing any of the configured negative
    cues, positive otherwise. Used by tests that exercise ``polarity_filter``
    without loading a real ONNX model.
    """

    NEGATIVE_CUES = {
        "burned out", "burnout", "micromanage", "toxic", "quitting",
        "frustrated", "lawyer", "hate", "miserable", "terrible",
        "harassment", "bullying", "discrimination",
    }

    POSITIVE_CUES = {
        "great", "love", "excellent", "supportive", "wonderful",
        "amazing", "happy", "preventing", "thrive",
    }

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return self._neutral()
        return self.classify_batch([text])[0]

    def classify_batch(self, texts):
        out = []
        for t in texts:
            if not t or not t.strip():
                out.append(self._neutral())
                continue
            lower = t.lower()
            neg = sum(cue in lower for cue in self.NEGATIVE_CUES)
            pos = sum(cue in lower for cue in self.POSITIVE_CUES)
            if neg > pos:
                out.append({
                    "label": "NEGATIVE",
                    "score": min(0.5 + 0.1 * neg, 0.99),
                    "scores": {"negative": min(0.5 + 0.1 * neg, 0.99),
                               "positive": max(1 - (0.5 + 0.1 * neg), 0.01)},
                })
            elif pos > neg:
                out.append({
                    "label": "POSITIVE",
                    "score": min(0.5 + 0.1 * pos, 0.99),
                    "scores": {"positive": min(0.5 + 0.1 * pos, 0.99),
                               "negative": max(1 - (0.5 + 0.1 * pos), 0.01)},
                })
            else:
                out.append(self._neutral())
        return out

    def polarity_score(self, text: str) -> float:
        r = self.classify(text)
        return float(r["scores"]["positive"] - r["scores"]["negative"])

    def polarity_scores(self, texts):
        results = self.classify_batch(texts)
        return [
            float(r["scores"]["positive"] - r["scores"]["negative"])
            for r in results
        ]

    def filter_negative(self, sentences, threshold: float = 0.0):
        scores = self.polarity_scores(sentences)
        return [s for s, sc in zip(sentences, scores) if sc < threshold]

    @staticmethod
    def split_sentences(doc: str):
        if not doc or not doc.strip():
            return []
        import re
        parts = re.split(r"(?<=[.!?])\s+", doc.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _neutral():
        return {"label": "NEUTRAL", "score": 0.5,
                "scores": {"negative": 0.5, "positive": 0.5}}


@pytest.fixture
def fake_polarity():
    return FakePolarityClassifier()


@pytest.fixture
def patch_polarity(monkeypatch, fake_polarity):
    """Patch ``keyneg.polarity.get_polarity_classifier`` to return the fake."""
    import keyneg.polarity as polarity_mod
    monkeypatch.setattr(polarity_mod, "get_polarity_classifier",
                        lambda *a, **kw: fake_polarity)
    return fake_polarity
