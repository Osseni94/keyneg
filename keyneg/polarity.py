"""
KeyNeg - Polarity Classifier (optional extra)
==============================================
Real polarity classification using DistilBERT-SST2 in ONNX form.

This module is the answer to the methodology critique: cosine similarity
between a doc and a list of negative labels measures *topical overlap*,
not *polarity*. A document about "burnout prevention" topically matches
"burnout" but is positive in tone. By running a real classifier first,
the pipeline can filter on actual polarity before doing taxonomy matching.

Installation
------------
The polarity dependencies are an optional extra::

    pip install keyneg[polarity]

This pulls in ``onnxruntime`` and ``transformers`` (for the tokenizer).
On first use the model is downloaded from HuggingFace and cached under
``~/.cache/keyneg/sentiment/``. Subsequent runs load from disk.

If the extras are not installed, ``PolarityClassifier()`` raises
``PolarityError`` with installation instructions; ``analyze()`` with
``polarity_filter=True`` propagates the error.

For air-gapped use install the ``keyneg-enterprise`` package, which
bundles the model in the wheel.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


DEFAULT_HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("KEYNEG_CACHE_DIR")
    or (Path.home() / ".cache" / "keyneg" / "sentiment")
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


class PolarityError(Exception):
    """Raised when the polarity classifier cannot be loaded or run."""


def _ensure_model_files(cache_dir: Path,
                        hf_model: str = DEFAULT_HF_MODEL) -> Path:
    """Download model files into ``cache_dir`` if missing. Returns the dir."""
    cache_dir = Path(cache_dir)
    onnx_path = cache_dir / "model.onnx"
    tokenizer_path = cache_dir / "tokenizer.json"
    config_path = cache_dir / "config.json"

    if onnx_path.exists() and tokenizer_path.exists() and config_path.exists():
        return cache_dir

    try:
        from transformers import AutoTokenizer, AutoConfig
        from transformers.onnx import export, FeaturesManager
    except ImportError as exc:
        raise PolarityError(
            "Polarity extras are not installed. Run: pip install keyneg[polarity]"
        ) from exc

    logger.info("Downloading polarity model %s into %s (one-time)", hf_model, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.save_pretrained(cache_dir)
    # transformers writes tokenizer.json when the tokenizer is a fast one;
    # force-save it explicitly if not already there.
    if not tokenizer_path.exists():
        # Fall back to writing a tokenizer.json from the slow tokenizer.
        # All HF distilbert checkpoints ship a fast tokenizer though.
        raise PolarityError(
            f"Tokenizer for {hf_model} does not provide a fast tokenizer.json; "
            "use a different HF checkpoint or supply --model_path manually."
        )

    config = AutoConfig.from_pretrained(hf_model)
    config.save_pretrained(cache_dir)

    # Export the model to ONNX.
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(hf_model)
    onnx_config_constructor = FeaturesManager.get_config(
        model.config.model_type, feature="sequence-classification"
    )
    onnx_config = onnx_config_constructor(model.config)
    export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=14,
        output=onnx_path,
    )

    return cache_dir


class PolarityClassifier:
    """Thin wrapper over an ONNX DistilBERT sentiment-classification head."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    DEFAULT_MAX_TOKENS = 512

    def __init__(self,
                 model_path: Optional[str] = None,
                 hf_model: str = DEFAULT_HF_MODEL,
                 cache_dir: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 verbose: bool = False):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise PolarityError(
                "Polarity classifier requires `onnxruntime` and `tokenizers`. "
                "Install with: pip install keyneg[polarity]"
            ) from exc

        if model_path:
            model_dir = Path(model_path)
        else:
            cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
            model_dir = _ensure_model_files(cache, hf_model)

        onnx_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "config.json"

        for required in (onnx_path, tokenizer_path, config_path):
            if not required.exists():
                raise PolarityError(f"Missing required polarity model file: {required}")

        self._verbose = verbose
        self._max_tokens = max_tokens

        if verbose:
            logger.info("Loading polarity model from %s", model_dir)

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_tokens)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self._id2label = {
            int(k): v.upper()
            for k, v in config.get("id2label", {"0": "NEGATIVE", "1": "POSITIVE"}).items()
        }

    # ---------------------------------------------------------------------
    # Public API (mirrors keyneg_enterprise.polarity.PolarityClassifier)
    # ---------------------------------------------------------------------

    def classify(self, text: str) -> Dict:
        if not text or not text.strip():
            return self._empty_result()
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [self._empty_result() for _ in texts]

        valid_texts = [texts[i].strip() for i in valid_indices]
        encoded = self._tokenizer.encode_batch(valid_texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        logits = self._session.run(None, ort_inputs)[0]
        probs = self._softmax(logits)

        results: List[Dict] = [self._empty_result() for _ in texts]
        for batch_idx, doc_idx in enumerate(valid_indices):
            results[doc_idx] = self._build_result(probs[batch_idx])
        return results

    def polarity_score(self, text: str) -> float:
        result = self.classify(text)
        scores = result["scores"]
        return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))

    def polarity_scores(self, texts: List[str]) -> List[float]:
        results = self.classify_batch(texts)
        return [
            float(r["scores"].get("positive", 0.0) - r["scores"].get("negative", 0.0))
            for r in results
        ]

    def filter_negative(self, sentences: List[str],
                        threshold: float = 0.0) -> List[str]:
        if not sentences:
            return []
        scores = self.polarity_scores(sentences)
        return [s for s, score in zip(sentences, scores) if score < threshold]

    @staticmethod
    def split_sentences(doc: str) -> List[str]:
        if not doc or not doc.strip():
            return []
        parts = _SENTENCE_SPLIT.split(doc.strip())
        return [p.strip() for p in parts if p.strip()]

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _build_result(self, probs: np.ndarray) -> Dict:
        scores = {self._id2label[i].lower(): float(p) for i, p in enumerate(probs)}
        top_idx = int(np.argmax(probs))
        return {
            "label": self._id2label[top_idx],
            "score": float(probs[top_idx]),
            "scores": scores,
        }

    def _empty_result(self) -> Dict:
        return {
            "label": "NEUTRAL",
            "score": 0.0,
            "scores": {"negative": 0.0, "positive": 0.0},
        }

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


_classifier_singleton: Optional[PolarityClassifier] = None


def get_polarity_classifier(
    model_path: Optional[str] = None,
    force_reload: bool = False,
    verbose: bool = False,
) -> PolarityClassifier:
    """Return a process-wide cached PolarityClassifier instance."""
    global _classifier_singleton
    if force_reload or _classifier_singleton is None:
        _classifier_singleton = PolarityClassifier(
            model_path=model_path, verbose=verbose
        )
    return _classifier_singleton
