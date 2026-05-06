"""
KeyNeg Core Module
==================
Main KeyNeg class for negative keyword and sentiment extraction.
Inspired by KeyBERT's clean API design.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import logging
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from .taxonomy import SENTIMENT_LABELS, NEGATIVE_TAXONOMY
from .negation import find_unnegated_matches

logger = logging.getLogger(__name__)


class KeyNeg:
    """
    KeyNeg: A KeyBERT-style negative sentiment and keyword extractor.

    Extracts negative keywords, frustration indicators, and discontent signals
    from text. Designed for workforce intelligence and marketing analysis.

    Usage:
        >>> from keyneg import KeyNeg
        >>> kn = KeyNeg()
        >>> keywords = kn.extract_keywords("I'm frustrated with the micromanagement")
        >>> sentiments = kn.extract_sentiments("The toxic culture is unbearable")
    """

    def __init__(
        self,
        model: Union[str, SentenceTransformer] = "all-mpnet-base-v2",
        custom_labels: Optional[List[str]] = None,
        custom_taxonomy: Optional[Dict] = None,
        extra_negation_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize KeyNeg.

        Args:
            model: SentenceTransformer model name or instance.
                   Default is 'all-mpnet-base-v2' for best performance.
            custom_labels: Optional list of custom sentiment labels to use
                          instead of or in addition to defaults.
            custom_taxonomy: Optional custom taxonomy dictionary to merge with
                            or replace the default taxonomy.
            extra_negation_tokens: Domain-specific negators added on top of
                          the built-in set. Useful for legal/regulatory
                          idioms (e.g., ['notwithstanding']).
        """
        # Load or use provided model
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
            self.model_name = model
        else:
            self.model = model
            self.model_name = "custom"

        # Setup labels
        self.labels = custom_labels if custom_labels else SENTIMENT_LABELS.copy()

        # Setup taxonomy (deepcopy so per-instance mutations don't leak into
        # the module-level NEGATIVE_TAXONOMY dict shared across instances).
        self.taxonomy = deepcopy(NEGATIVE_TAXONOMY)
        if custom_taxonomy:
            self._merge_taxonomy(custom_taxonomy)

        # Domain-specific negation cues forwarded to find_unnegated_matches.
        self._extra_negation_tokens: Optional[List[str]] = (
            list(extra_negation_tokens) if extra_negation_tokens else None
        )

        # Pre-compute label embeddings
        self._label_embeddings = None
        self._keyword_embeddings = None
        self._all_keywords = None
        self._all_keywords_lower = None

    def _merge_taxonomy(self, custom: Dict):
        """Merge custom taxonomy with default."""
        for key, value in custom.items():
            if key in self.taxonomy and isinstance(value, dict):
                self.taxonomy[key].update(value)
            else:
                self.taxonomy[key] = value

    @property
    def label_embeddings(self) -> np.ndarray:
        """Lazily compute and cache label embeddings."""
        if self._label_embeddings is None:
            self._label_embeddings = self.model.encode(
                self.labels, show_progress_bar=False
            )
        return self._label_embeddings

    @property
    def all_keywords(self) -> List[str]:
        """Flat list of unique keywords from this instance's taxonomy.

        Reads from ``self.taxonomy`` (not the module-level constant) so that
        ``add_custom_keywords`` actually surfaces in extraction. Pre-1.2 this
        called the module-level ``get_all_keywords()``, which silently
        ignored per-instance customizations.
        """
        if self._all_keywords is None:
            keywords: List[str] = []

            def _walk(node):
                if isinstance(node, list):
                    keywords.extend(k for k in node if isinstance(k, str))
                elif isinstance(node, dict):
                    for value in node.values():
                        _walk(value)

            _walk(self.taxonomy)
            # Dedupe while preserving order so MMR / batching are stable.
            seen = set()
            unique: List[str] = []
            for kw in keywords:
                if kw not in seen:
                    seen.add(kw)
                    unique.append(kw)
            self._all_keywords = unique
        return self._all_keywords

    @property
    def all_keywords_lower(self) -> set:
        """Lowercased set of taxonomy keywords for O(1) membership tests."""
        if self._all_keywords_lower is None:
            self._all_keywords_lower = {k.lower() for k in self.all_keywords}
        return self._all_keywords_lower

    @property
    def keyword_embeddings(self) -> np.ndarray:
        """Lazily compute and cache keyword embeddings."""
        if self._keyword_embeddings is None:
            self._keyword_embeddings = self.model.encode(
                self.all_keywords, show_progress_bar=False
            )
        return self._keyword_embeddings

    def extract_sentiments(
        self,
        doc: str,
        top_n: int = 5,
        threshold: float = 0.3,
        diversity: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Extract top negative sentiment labels from a document.

        This is the primary method for workforce intelligence analysis.
        It matches the document against predefined sentiment categories.

        Args:
            doc: Input text to analyze.
            top_n: Number of top sentiments to return.
            threshold: Minimum similarity score (0-1) to include.
            diversity: MMR diversity parameter (0-1). Higher = more diverse results.

        Returns:
            List of (sentiment_label, score) tuples sorted by score descending.

        Example:
            >>> kn = KeyNeg()
            >>> sentiments = kn.extract_sentiments(
            ...     "My manager micromanages everything and never listens"
            ... )
            >>> print(sentiments[0])
            ('micromanagement', 0.72)
        """
        if not doc or not doc.strip():
            return []

        # Encode document
        doc_embedding = self.model.encode([doc.strip()], show_progress_bar=False)[0]

        # Compute similarities
        similarities = cosine_similarity([doc_embedding], self.label_embeddings)[0]

        # Create results
        results = list(zip(self.labels, similarities))
        results = [(label, float(score)) for label, score in results if score >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        if diversity > 0 and len(results) > top_n:
            # Apply MMR for diversity
            results = self._mmr_diversify(
                doc_embedding, results, top_n, diversity
            )

        return results[:top_n]

    def extract_keywords(
        self,
        doc: str,
        top_n: int = 10,
        threshold: float = 0.25,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        use_taxonomy: bool = True,
        diversity: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Extract negative keywords from a document.

        This method extracts specific negative keywords and phrases,
        matching against both the taxonomy and document-derived candidates.

        Args:
            doc: Input text to analyze.
            top_n: Number of keywords to return.
            threshold: Minimum similarity score to include.
            keyphrase_ngram_range: Range of n-grams to extract from document.
            use_taxonomy: Whether to match against taxonomy keywords.
            diversity: MMR diversity (0-1). Higher = more diverse.

        Returns:
            List of (keyword, score) tuples.

        Example:
            >>> kn = KeyNeg()
            >>> keywords = kn.extract_keywords(
            ...     "The constant micromanagement and lack of recognition is frustrating"
            ... )
        """
        if not doc or not doc.strip():
            return []

        doc = doc.strip()

        # Encode document
        doc_embedding = self.model.encode([doc], show_progress_bar=False)[0]

        all_candidates = []

        # Match against taxonomy keywords
        if use_taxonomy:
            similarities = cosine_similarity(
                [doc_embedding], self.keyword_embeddings
            )[0]
            for keyword, score in zip(self.all_keywords, similarities):
                if score >= threshold:
                    all_candidates.append((keyword, float(score)))

        # Extract candidates from document itself
        try:
            doc_candidates = self._extract_candidates(doc, keyphrase_ngram_range)
            if doc_candidates:
                candidate_embeddings = self.model.encode(
                    doc_candidates, show_progress_bar=False
                )
                similarities = cosine_similarity(
                    [doc_embedding], candidate_embeddings
                )[0]
                for candidate, score in zip(doc_candidates, similarities):
                    # Boost candidates that appear in taxonomy, but cap the
                    # boosted score at 1.0 so it remains a valid cosine sim.
                    boost = 1.2 if candidate.lower() in self.all_keywords_lower else 1.0
                    boosted = min(float(score) * boost, 1.0)
                    if boosted >= threshold:
                        all_candidates.append((candidate, boosted))
        except Exception as exc:
            logger.warning("n-gram extraction failed, falling back to taxonomy-only: %s", exc)

        # Deduplicate and sort
        seen = set()
        unique_candidates = []
        for kw, score in sorted(all_candidates, key=lambda x: x[1], reverse=True):
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_candidates.append((kw, score))

        if diversity > 0 and len(unique_candidates) > top_n:
            unique_candidates = self._mmr_diversify(
                doc_embedding, unique_candidates, top_n, diversity
            )

        return unique_candidates[:top_n]

    def extract_keywords_batch(
        self,
        docs: List[str],
        top_n: int = 10,
        threshold: float = 0.25,
        use_taxonomy: bool = True,
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        """
        Extract keywords from multiple documents efficiently.

        Args:
            docs: List of documents to analyze.
            top_n: Number of keywords per document.
            threshold: Minimum similarity threshold.
            use_taxonomy: Whether to use taxonomy matching.
            show_progress: Show progress bar.

        Returns:
            List of keyword lists, one per document.
        """
        if not docs:
            return []

        # Clean docs
        cleaned_docs = [d.strip() if d else "" for d in docs]
        valid_indices = [i for i, d in enumerate(cleaned_docs) if d]

        if not valid_indices:
            return [[] for _ in docs]

        valid_docs = [cleaned_docs[i] for i in valid_indices]

        # Batch encode
        doc_embeddings = self.model.encode(
            valid_docs, show_progress_bar=show_progress
        )

        results = [[] for _ in docs]

        if use_taxonomy:
            # Compute all similarities at once
            all_similarities = cosine_similarity(doc_embeddings, self.keyword_embeddings)

            for idx, doc_idx in enumerate(valid_indices):
                similarities = all_similarities[idx]
                candidates = [
                    (self.all_keywords[i], float(similarities[i]))
                    for i in range(len(self.all_keywords))
                    if similarities[i] >= threshold
                ]
                candidates.sort(key=lambda x: x[1], reverse=True)
                results[doc_idx] = candidates[:top_n]

        return results

    def extract_sentiments_batch(
        self,
        docs: List[str],
        top_n: int = 5,
        threshold: float = 0.3,
        show_progress: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        """
        Extract sentiments from multiple documents efficiently.

        Args:
            docs: List of documents to analyze.
            top_n: Number of sentiments per document.
            threshold: Minimum similarity threshold.
            show_progress: Show progress bar.

        Returns:
            List of sentiment lists, one per document.
        """
        if not docs:
            return []

        # Clean docs
        cleaned_docs = [d.strip() if d else "" for d in docs]
        valid_indices = [i for i, d in enumerate(cleaned_docs) if d]

        if not valid_indices:
            return [[] for _ in docs]

        valid_docs = [cleaned_docs[i] for i in valid_indices]

        # Batch encode
        doc_embeddings = self.model.encode(
            valid_docs, show_progress_bar=show_progress
        )

        # Compute all similarities at once
        all_similarities = cosine_similarity(doc_embeddings, self.label_embeddings)

        results = [[] for _ in docs]
        for idx, doc_idx in enumerate(valid_indices):
            similarities = all_similarities[idx]
            sentiments = [
                (self.labels[i], float(similarities[i]))
                for i in range(len(self.labels))
                if similarities[i] >= threshold
            ]
            sentiments.sort(key=lambda x: x[1], reverse=True)
            results[doc_idx] = sentiments[:top_n]

        return results

    def analyze(
        self,
        doc: str,
        top_n_keywords: int = 10,
        top_n_sentiments: int = 5,
        keyword_threshold: float = 0.25,
        sentiment_threshold: float = 0.3,
        polarity_filter: bool = False,
        polarity_threshold: float = 0.0,
    ) -> Dict:
        """
        Comprehensive analysis of a document.

        Args:
            doc: Input text.
            top_n_keywords: Number of keywords to extract.
            top_n_sentiments: Number of sentiments to extract.
            keyword_threshold: Threshold for keywords.
            sentiment_threshold: Threshold for sentiments.
            polarity_filter: If True, run a real polarity classifier first
                and restrict keyword/sentiment extraction to sentences whose
                polarity is below ``polarity_threshold``. Requires the
                ``polarity`` extra: ``pip install keyneg[polarity]``.
            polarity_threshold: Sentences with ``polarity_score < threshold``
                are kept. Default 0.0 keeps any net-negative sentence; raise
                it (e.g. -0.3) to keep only strongly negative sentences.

        Returns:
            Dictionary with:

            - ``keywords`` / ``sentiments`` / ``top_sentiment`` / ``categories``
            - ``topic_match_score``: mean cosine similarity to detected
              negative labels in [0, 1]. Measures *topical overlap* with
              negative themes — not polarity.
            - ``negativity_score``: alias for ``topic_match_score`` retained
              for backward compatibility.
            - ``polarity_score``: signed polarity in [-1, 1] (negative =
              negative tone). Populated only when ``polarity_filter=True``;
              defaults to 0.0 otherwise.
            - ``polarity_filter_applied``: True if filtering ran.
            - ``negative_sentences``: list of sentences kept by the polarity
              filter (empty unless ``polarity_filter=True``).

        Example:
            >>> result = kn.analyze("I hate the toxic culture here")
            >>> print(result['top_sentiment'])
            'toxic culture'
            >>> print(result['topic_match_score'])
            0.65
        """
        empty_result = {
            "keywords": [],
            "sentiments": [],
            "top_sentiment": None,
            "topic_match_score": 0.0,
            "negativity_score": 0.0,
            "polarity_score": 0.0,
            "polarity_filter_applied": False,
            "negative_sentences": [],
            "categories": [],
        }

        if not doc or not doc.strip():
            return empty_result

        analysis_doc = doc
        polarity_score = 0.0
        negative_sentences: List[str] = []
        polarity_applied = False

        if polarity_filter:
            # Lazy import: the polarity classifier requires the optional
            # ``polarity`` extra (onnxruntime, transformers). We import here
            # so users who never call ``polarity_filter=True`` don't pay the
            # import cost — and so a missing extra fails with a clear message
            # only when actually requested.
            from .polarity import get_polarity_classifier
            classifier = get_polarity_classifier()
            sentences = classifier.split_sentences(doc)
            if sentences:
                sentence_scores = classifier.polarity_scores(sentences)
                negative_sentences = [
                    s for s, score in zip(sentences, sentence_scores)
                    if score < polarity_threshold
                ]
                polarity_score = float(np.mean(sentence_scores))
                polarity_applied = True

                if not negative_sentences:
                    # Document scanned, nothing was negative-leaning.
                    return {
                        **empty_result,
                        "polarity_score": polarity_score,
                        "polarity_filter_applied": True,
                    }
                analysis_doc = " ".join(negative_sentences)

        keywords = self.extract_keywords(
            analysis_doc, top_n=top_n_keywords, threshold=keyword_threshold
        )
        sentiments = self.extract_sentiments(
            analysis_doc, top_n=top_n_sentiments, threshold=sentiment_threshold
        )

        topic_match = float(np.mean([s[1] for s in sentiments])) if sentiments else 0.0
        categories = self._identify_categories(keywords)

        return {
            "keywords": keywords,
            "sentiments": sentiments,
            "top_sentiment": sentiments[0][0] if sentiments else None,
            "topic_match_score": topic_match,
            # Alias kept for backward compatibility with v1.1.x callers.
            # New code should read ``topic_match_score`` (and ``polarity_score``
            # when ``polarity_filter=True``).
            "negativity_score": topic_match,
            "polarity_score": polarity_score,
            "polarity_filter_applied": polarity_applied,
            "negative_sentences": negative_sentences,
            "categories": categories,
        }

    def analyze_batch(
        self,
        docs: List[str],
        top_n_keywords: int = 10,
        top_n_sentiments: int = 5,
        show_progress: bool = True,
        polarity_filter: bool = False,
        polarity_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Batch analysis of multiple documents.

        Args:
            docs: List of documents.
            top_n_keywords / top_n_sentiments / show_progress: as for the
                non-batch methods.
            polarity_filter / polarity_threshold: see ``analyze``.

        Returns:
            List of analysis dictionaries. See ``analyze`` for shape.
        """
        if polarity_filter:
            # Per-doc routing; the polarity classifier is called once per
            # doc internally so we don't lose batch parallelism for the
            # post-filter taxonomy step.
            return [
                self.analyze(
                    doc,
                    top_n_keywords=top_n_keywords,
                    top_n_sentiments=top_n_sentiments,
                    polarity_filter=True,
                    polarity_threshold=polarity_threshold,
                )
                for doc in docs
            ]

        keywords_batch = self.extract_keywords_batch(
            docs, top_n=top_n_keywords, show_progress=show_progress
        )
        sentiments_batch = self.extract_sentiments_batch(
            docs, top_n=top_n_sentiments, show_progress=show_progress
        )

        results: List[Dict] = []
        for keywords, sentiments in zip(keywords_batch, sentiments_batch):
            topic_match = float(np.mean([s[1] for s in sentiments])) if sentiments else 0.0
            categories = self._identify_categories(keywords)
            results.append({
                "keywords": keywords,
                "sentiments": sentiments,
                "top_sentiment": sentiments[0][0] if sentiments else None,
                "topic_match_score": topic_match,
                "negativity_score": topic_match,
                "polarity_score": 0.0,
                "polarity_filter_applied": False,
                "negative_sentences": [],
                "categories": categories,
            })

        return results

    def get_intensity(self, doc: str) -> Dict:
        """
        Analyze the intensity level of negativity in text.

        Negated mentions are skipped: "I'm not absolutely furious" does
        not register the "absolutely" intensifier.

        Returns:
            Dictionary with 'level' (1-4), 'label', and 'indicators'.
        """
        if not doc:
            return {"level": 0, "label": "neutral", "indicators": []}

        intensity_keywords = self.taxonomy.get("emotional_states", {}).get(
            "intensity_expressions", {}
        )

        levels = {"mild": 1, "moderate": 2, "strong": 3, "extreme": 4}

        found_level = 0
        found_label = "neutral"
        found_indicators: List[str] = []

        for label, level in levels.items():
            keywords = intensity_keywords.get(label, [])
            matches = find_unnegated_matches(
                doc, keywords,
                extra_negation_tokens=self._extra_negation_tokens,
            )
            if matches and level > found_level:
                found_level = level
                found_label = label
                found_indicators = matches

        return {
            "level": found_level,
            "label": found_label,
            "indicators": found_indicators,
        }

    def detect_departure_intent(self, doc: str) -> Dict:
        """
        Detect signals of intent to leave/quit.

        Negated mentions are skipped: "I'm not quitting" returns no signals.

        Returns:
            Dictionary with 'detected', 'confidence', and 'signals'.
        """
        if not doc:
            return {"detected": False, "confidence": 0.0, "signals": []}

        departure_keywords = self.taxonomy.get("action_indicators", {}).get(
            "departure_intent", []
        )
        matches = find_unnegated_matches(
            doc, departure_keywords,
            extra_negation_tokens=self._extra_negation_tokens,
        )
        confidence = min(len(matches) / 3.0, 1.0)

        return {
            "detected": len(matches) > 0,
            "confidence": confidence,
            "signals": matches,
        }

    def detect_escalation_risk(self, doc: str) -> Dict:
        """
        Detect signals of escalation (legal threats, going public, etc.).

        Negated mentions are skipped: "I'm not contacting any lawyer"
        returns no signals.

        Returns:
            Dictionary with 'detected', 'risk_level', and 'signals'.
        """
        if not doc:
            return {"detected": False, "risk_level": "low", "signals": []}

        escalation_keywords = self.taxonomy.get("action_indicators", {}).get(
            "escalation_threats", []
        )
        matches = find_unnegated_matches(
            doc, escalation_keywords,
            extra_negation_tokens=self._extra_negation_tokens,
        )

        if len(matches) >= 3:
            risk_level = "high"
        elif len(matches) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "detected": len(matches) > 0,
            "risk_level": risk_level,
            "signals": matches,
        }

    def _extract_candidates(
        self, doc: str, ngram_range: Tuple[int, int]
    ) -> List[str]:
        """Extract n-gram candidates from document."""
        try:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                stop_words="english",
                max_features=100,
            )
            vectorizer.fit([doc])
            return list(vectorizer.get_feature_names_out())
        except ValueError as exc:
            # CountVectorizer raises ValueError for empty/stop-word-only docs.
            logger.debug("CountVectorizer found no candidates for doc: %s", exc)
            return []

    def _mmr_diversify(
        self,
        doc_embedding: np.ndarray,
        candidates: List[Tuple[str, float]],
        top_n: int,
        diversity: float,
    ) -> List[Tuple[str, float]]:
        """Apply Maximal Marginal Relevance for diversity."""
        if len(candidates) <= top_n:
            return candidates

        # Get embeddings for candidates
        candidate_texts = [c[0] for c in candidates]
        candidate_embeddings = self.model.encode(
            candidate_texts, show_progress_bar=False
        )

        # Start with highest scored
        selected = [0]
        selected_embeddings = [candidate_embeddings[0]]

        while len(selected) < top_n:
            best_score = float("-inf")
            best_idx = -1

            for i in range(len(candidates)):
                if i in selected:
                    continue

                # Relevance to document
                relevance = candidates[i][1]

                # Max similarity to already selected
                sims = cosine_similarity(
                    [candidate_embeddings[i]], selected_embeddings
                )[0]
                max_sim = max(sims)

                # MMR score
                mmr_score = (1 - diversity) * relevance - diversity * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])

        return [candidates[i] for i in selected]

    def _identify_categories(
        self, keywords: List[Tuple[str, float]]
    ) -> List[str]:
        """Identify taxonomy categories for given keywords."""
        categories = set()
        keyword_texts = {kw[0].lower() for kw in keywords}

        for category, subcategories in self.taxonomy.items():
            if isinstance(subcategories, dict):
                for subcat, kws in subcategories.items():
                    if isinstance(kws, list):
                        if any(kw.lower() in keyword_texts for kw in kws):
                            categories.add(category)
                            break
                    elif isinstance(kws, dict):
                        for subsubkws in kws.values():
                            if isinstance(subsubkws, list):
                                if any(kw.lower() in keyword_texts for kw in subsubkws):
                                    categories.add(category)
                                    break

        return list(categories)

    def summarize_by_label(
        self,
        docs: List[str],
        top_n: int = 3,
        examples_per_label: int = 3,
        threshold: float = 0.3,
        show_progress: bool = True,
    ) -> Dict:
        """
        Analyze multiple documents and group them by sentiment label.

        Takes a batch of texts, analyzes each for sentiment, and returns
        a summary grouped by label with example quotes for each complaint type.
        Perfect for generating reports from customer feedback or reviews.

        Args:
            docs: List of documents to analyze and group.
            top_n: Number of sentiment labels to consider per document (default: 3).
            examples_per_label: Max example quotes per label (default: 3).
            threshold: Minimum similarity threshold (default: 0.3).
            show_progress: Show progress bar during embedding (default: True).

        Returns:
            Dictionary with:
            - total_docs: Number of documents processed
            - unique_labels: Number of unique labels found
            - summary: Dict mapping label -> {count, avg_score, examples}

        Example:
            >>> kn = KeyNeg()
            >>> result = kn.summarize_by_label([
            ...     "The service was terrible",
            ...     "Staff was rude and unhelpful",
            ...     "Billing department never responds",
            ... ])
            >>> print(result['summary']['poor customer service'])
            {'count': 2, 'avg_score': 0.65, 'examples': [...]}
        """
        if not docs:
            return {
                "total_docs": 0,
                "unique_labels": 0,
                "summary": {},
            }

        # Get sentiments for all docs
        sentiments_batch = self.extract_sentiments_batch(
            docs,
            top_n=top_n,
            threshold=threshold,
            show_progress=show_progress,
        )

        # Group by label
        label_groups: Dict[str, Dict] = {}

        for doc, sentiments in zip(docs, sentiments_batch):
            if not sentiments:
                continue

            for label, score in sentiments:
                if label not in label_groups:
                    label_groups[label] = {
                        "count": 0,
                        "total_score": 0.0,
                        "examples": [],
                    }

                label_groups[label]["count"] += 1
                label_groups[label]["total_score"] += score

                # Store example with score
                if len(label_groups[label]["examples"]) < examples_per_label:
                    truncated = doc[:150] + "..." if len(doc) > 150 else doc
                    label_groups[label]["examples"].append({
                        "text": truncated,
                        "score": round(score, 4),
                    })

        # Format output - sort by count descending
        summary = {}
        for label, data in sorted(label_groups.items(), key=lambda x: -x[1]["count"]):
            avg_score = data["total_score"] / data["count"] if data["count"] > 0 else 0
            summary[label] = {
                "count": data["count"],
                "avg_score": round(avg_score, 4),
                "examples": data["examples"],
            }

        return {
            "total_docs": len(docs),
            "unique_labels": len(summary),
            "summary": summary,
        }

    def add_custom_labels(self, labels: List[str]):
        """Add custom sentiment labels."""
        self.labels.extend(labels)
        self._label_embeddings = None  # Reset cache

    def add_custom_keywords(self, category: str, keywords: List[str]):
        """Add custom keywords to a taxonomy category."""
        if category not in self.taxonomy:
            self.taxonomy[category] = {"custom": list(keywords)}
        elif isinstance(self.taxonomy[category], dict):
            if "custom" in self.taxonomy[category]:
                self.taxonomy[category]["custom"].extend(keywords)
            else:
                self.taxonomy[category]["custom"] = list(keywords)
        # Reset all keyword-derived caches.
        self._all_keywords = None
        self._all_keywords_lower = None
        self._keyword_embeddings = None

    def __repr__(self):
        return f"KeyNeg(model='{self.model_name}', labels={len(self.labels)}, keywords={len(self.all_keywords)})"
