"""
KeyNeg Core Module
==================
Main KeyNeg class for negative keyword and sentiment extraction.
Inspired by KeyBERT's clean API design.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings

from .taxonomy import SENTIMENT_LABELS, NEGATIVE_TAXONOMY, get_all_keywords


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

        # Setup taxonomy
        self.taxonomy = NEGATIVE_TAXONOMY.copy()
        if custom_taxonomy:
            self._merge_taxonomy(custom_taxonomy)

        # Pre-compute label embeddings
        self._label_embeddings = None
        self._keyword_embeddings = None
        self._all_keywords = None

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
        """Get all keywords from taxonomy."""
        if self._all_keywords is None:
            self._all_keywords = get_all_keywords()
        return self._all_keywords

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
                    # Boost candidates that appear in taxonomy
                    boost = 1.2 if candidate.lower() in [k.lower() for k in self.all_keywords] else 1.0
                    if score * boost >= threshold:
                        all_candidates.append((candidate, float(score * boost)))
        except Exception:
            pass  # Fall back to taxonomy-only if extraction fails

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
    ) -> Dict:
        """
        Comprehensive analysis of a document.

        Extracts both keywords and sentiments in a single call.

        Args:
            doc: Input text.
            top_n_keywords: Number of keywords to extract.
            top_n_sentiments: Number of sentiments to extract.
            keyword_threshold: Threshold for keywords.
            sentiment_threshold: Threshold for sentiments.

        Returns:
            Dictionary with 'keywords', 'sentiments', 'top_sentiment',
            'negativity_score', and 'categories'.

        Example:
            >>> result = kn.analyze("I hate the toxic culture here")
            >>> print(result['top_sentiment'])
            'toxic culture'
        """
        if not doc or not doc.strip():
            return {
                "keywords": [],
                "sentiments": [],
                "top_sentiment": None,
                "negativity_score": 0.0,
                "categories": [],
            }

        keywords = self.extract_keywords(
            doc, top_n=top_n_keywords, threshold=keyword_threshold
        )
        sentiments = self.extract_sentiments(
            doc, top_n=top_n_sentiments, threshold=sentiment_threshold
        )

        # Calculate overall negativity score
        negativity_score = np.mean([s[1] for s in sentiments]) if sentiments else 0.0

        # Identify categories
        categories = self._identify_categories(keywords)

        return {
            "keywords": keywords,
            "sentiments": sentiments,
            "top_sentiment": sentiments[0][0] if sentiments else None,
            "negativity_score": float(negativity_score),
            "categories": categories,
        }

    def analyze_batch(
        self,
        docs: List[str],
        top_n_keywords: int = 10,
        top_n_sentiments: int = 5,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Batch analysis of multiple documents.

        Args:
            docs: List of documents.
            top_n_keywords: Keywords per document.
            top_n_sentiments: Sentiments per document.
            show_progress: Show progress bar.

        Returns:
            List of analysis dictionaries.
        """
        keywords_batch = self.extract_keywords_batch(
            docs, top_n=top_n_keywords, show_progress=show_progress
        )
        sentiments_batch = self.extract_sentiments_batch(
            docs, top_n=top_n_sentiments, show_progress=show_progress
        )

        results = []
        for keywords, sentiments in zip(keywords_batch, sentiments_batch):
            negativity_score = np.mean([s[1] for s in sentiments]) if sentiments else 0.0
            categories = self._identify_categories(keywords)
            results.append({
                "keywords": keywords,
                "sentiments": sentiments,
                "top_sentiment": sentiments[0][0] if sentiments else None,
                "negativity_score": float(negativity_score),
                "categories": categories,
            })

        return results

    def get_intensity(self, doc: str) -> Dict:
        """
        Analyze the intensity level of negativity in text.

        Returns:
            Dictionary with 'level' (1-4), 'label', and 'indicators'.
        """
        intensity_keywords = self.taxonomy.get("emotional_states", {}).get(
            "intensity_expressions", {}
        )

        doc_lower = doc.lower()

        # Check each intensity level
        levels = {
            "mild": 1,
            "moderate": 2,
            "strong": 3,
            "extreme": 4,
        }

        found_level = 0
        found_label = "neutral"
        found_indicators = []

        for label, level in levels.items():
            keywords = intensity_keywords.get(label, [])
            matches = [kw for kw in keywords if kw.lower() in doc_lower]
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

        Returns:
            Dictionary with 'detected', 'confidence', and 'signals'.
        """
        departure_keywords = self.taxonomy.get("action_indicators", {}).get(
            "departure_intent", []
        )

        doc_lower = doc.lower()
        matches = [kw for kw in departure_keywords if kw.lower() in doc_lower]

        confidence = min(len(matches) / 3.0, 1.0)  # Cap at 1.0

        return {
            "detected": len(matches) > 0,
            "confidence": confidence,
            "signals": matches,
        }

    def detect_escalation_risk(self, doc: str) -> Dict:
        """
        Detect signals of escalation (legal threats, going public, etc.).

        Returns:
            Dictionary with 'detected', 'risk_level', and 'signals'.
        """
        escalation_keywords = self.taxonomy.get("action_indicators", {}).get(
            "escalation_threats", []
        )

        doc_lower = doc.lower()
        matches = [kw for kw in escalation_keywords if kw.lower() in doc_lower]

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
        except Exception:
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
            self.taxonomy[category] = {"custom": keywords}
        elif isinstance(self.taxonomy[category], dict):
            if "custom" in self.taxonomy[category]:
                self.taxonomy[category]["custom"].extend(keywords)
            else:
                self.taxonomy[category]["custom"] = keywords
        self._all_keywords = None  # Reset cache
        self._keyword_embeddings = None

    def __repr__(self):
        return f"KeyNeg(model='{self.model_name}', labels={len(self.labels)}, keywords={len(self.all_keywords)})"
