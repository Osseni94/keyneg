"""
KeyNeg: A KeyBERT-style Negative Sentiment and Keyword Extractor
================================================================

KeyNeg is designed for workforce intelligence and marketing analysis,
extracting negative sentiment, frustration indicators, and discontent
signals from text data.

Usage:
    >>> from keyneg import KeyNeg
    >>> kn = KeyNeg()
    >>>
    >>> # Extract negative sentiments
    >>> sentiments = kn.extract_sentiments("I'm frustrated with the micromanagement")
    >>> print(sentiments)
    [('micromanagement', 0.72), ('frustration', 0.68), ...]
    >>>
    >>> # Extract negative keywords
    >>> keywords = kn.extract_keywords("The toxic culture and burnout is unbearable")
    >>> print(keywords)
    [('toxic culture', 0.81), ('burnout', 0.75), ...]
    >>>
    >>> # Full analysis
    >>> result = kn.analyze("My manager never listens and I'm thinking of quitting")
    >>> print(result['top_sentiment'])
    'poor leadership'
    >>> print(result['negativity_score'])
    0.65

For batch processing:
    >>> docs = ["Comment 1...", "Comment 2...", "Comment 3..."]
    >>> results = kn.analyze_batch(docs)

Summarize by label (group texts by complaint type):
    >>> docs = ["Service was terrible", "Staff rude", "Billing never responds"]
    >>> summary = kn.summarize_by_label(docs)
    >>> print(summary['summary']['poor customer service'])
    {'count': 2, 'avg_score': 0.65, 'examples': [...]}

Special detectors:
    >>> kn.detect_departure_intent("I'm updating my resume and interviewing")
    {'detected': True, 'confidence': 0.67, 'signals': ['updating resume', 'interviewing']}

    >>> kn.detect_escalation_risk("I'm going to contact my lawyer about this")
    {'detected': True, 'risk_level': 'high', 'signals': ['contact my lawyer']}

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

__version__ = "1.1.0"
__author__ = "Kaossara Osseni"
__email__ = "admin@grandnasser.com"

from keyneg._keyneg import KeyNeg
from keyneg.taxonomy import (
    SENTIMENT_LABELS,
    NEGATIVE_TAXONOMY,
    get_all_keywords,
    get_keywords_by_category,
    get_category_labels,
)

__all__ = [
    "KeyNeg",
    "SENTIMENT_LABELS",
    "NEGATIVE_TAXONOMY",
    "get_all_keywords",
    "get_keywords_by_category",
    "get_category_labels",
]
