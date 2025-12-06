"""
KeyNeg Utilities Module
=======================
Helper functions for text processing, visualization, and export.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import re
from typing import List, Dict, Tuple, Optional
import json


def highlight_keywords(
    text: str,
    keywords: List[Tuple[str, float]],
    format: str = "html",
    threshold: float = 0.0,
) -> str:
    """
    Highlight detected keywords in text.

    Args:
        text: Original text.
        keywords: List of (keyword, score) tuples.
        format: Output format - 'html', 'markdown', 'terminal', or 'plain'.
        threshold: Only highlight keywords with score >= threshold.

    Returns:
        Text with highlighted keywords.

    Example:
        >>> highlighted = highlight_keywords(
        ...     "I'm frustrated with the micromanagement",
        ...     [("frustrated", 0.8), ("micromanagement", 0.75)],
        ...     format="html"
        ... )
    """
    if not keywords:
        return text

    # Filter by threshold
    filtered = [(kw, score) for kw, score in keywords if score >= threshold]

    if not filtered:
        return text

    # Sort by length (longest first) to handle overlapping matches
    filtered.sort(key=lambda x: len(x[0]), reverse=True)

    result = text

    for keyword, score in filtered:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        if format == "html":
            # Color based on score intensity
            intensity = int(255 * (1 - score))
            color = f"rgb(255, {intensity}, {intensity})"
            replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="Score: {score:.2f}">\\g<0></span>'
        elif format == "markdown":
            replacement = f"**\\g<0>**"
        elif format == "terminal":
            # ANSI color codes
            replacement = f"\033[91m\\g<0>\033[0m"
        else:  # plain
            replacement = f"[\\g<0>]"

        result = pattern.sub(replacement, result)

    return result


def score_to_severity(score: float) -> str:
    """
    Convert a similarity score to a severity label.

    Args:
        score: Similarity score (0-1).

    Returns:
        Severity label: 'low', 'medium', 'high', or 'critical'.
    """
    if score >= 0.7:
        return "critical"
    elif score >= 0.5:
        return "high"
    elif score >= 0.3:
        return "medium"
    else:
        return "low"


def format_results_table(
    results: List[Tuple[str, float]],
    title: str = "Results",
    show_severity: bool = True,
) -> str:
    """
    Format results as a text table.

    Args:
        results: List of (label, score) tuples.
        title: Table title.
        show_severity: Include severity column.

    Returns:
        Formatted table string.
    """
    if not results:
        return f"{title}\n(No results)"

    lines = [title, "=" * len(title)]

    if show_severity:
        header = f"{'Label':<40} {'Score':>8} {'Severity':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        for label, score in results:
            severity = score_to_severity(score)
            lines.append(f"{label:<40} {score:>8.3f} {severity:>10}")
    else:
        header = f"{'Label':<40} {'Score':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for label, score in results:
            lines.append(f"{label:<40} {score:>8.3f}")

    return "\n".join(lines)


def export_to_json(
    analysis_result: Dict,
    filepath: Optional[str] = None,
    indent: int = 2,
) -> str:
    """
    Export analysis results to JSON.

    Args:
        analysis_result: Result from KeyNeg.analyze().
        filepath: Optional file path to save to.
        indent: JSON indentation.

    Returns:
        JSON string.
    """
    output = json.dumps(analysis_result, indent=indent, ensure_ascii=False)

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)

    return output


def export_batch_to_csv(
    results: List[Dict],
    filepath: str,
    include_keywords: bool = True,
    max_keywords: int = 5,
) -> None:
    """
    Export batch analysis results to CSV.

    Args:
        results: List of analysis results from KeyNeg.analyze_batch().
        filepath: Output CSV file path.
        include_keywords: Include top keywords column.
        max_keywords: Maximum keywords to include per row.
    """
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["index", "top_sentiment", "negativity_score", "categories"]
        if include_keywords:
            header.append("top_keywords")
        writer.writerow(header)

        # Data
        for i, result in enumerate(results):
            row = [
                i,
                result.get("top_sentiment", ""),
                f"{result.get('negativity_score', 0):.3f}",
                "; ".join(result.get("categories", [])),
            ]
            if include_keywords:
                keywords = result.get("keywords", [])[:max_keywords]
                kw_str = "; ".join([f"{kw}({score:.2f})" for kw, score in keywords])
                row.append(kw_str)
            writer.writerow(row)


def aggregate_batch_results(
    results: List[Dict],
) -> Dict:
    """
    Aggregate batch results for summary statistics.

    Args:
        results: List of analysis results.

    Returns:
        Dictionary with aggregated statistics.
    """
    if not results:
        return {}

    # Collect all sentiments and keywords
    all_sentiments = {}
    all_keywords = {}
    all_categories = {}
    negativity_scores = []

    for result in results:
        negativity_scores.append(result.get("negativity_score", 0))

        for sentiment, score in result.get("sentiments", []):
            if sentiment not in all_sentiments:
                all_sentiments[sentiment] = {"count": 0, "total_score": 0}
            all_sentiments[sentiment]["count"] += 1
            all_sentiments[sentiment]["total_score"] += score

        for keyword, score in result.get("keywords", []):
            if keyword not in all_keywords:
                all_keywords[keyword] = {"count": 0, "total_score": 0}
            all_keywords[keyword]["count"] += 1
            all_keywords[keyword]["total_score"] += score

        for category in result.get("categories", []):
            all_categories[category] = all_categories.get(category, 0) + 1

    # Calculate averages and sort
    sentiment_summary = [
        {
            "sentiment": s,
            "count": data["count"],
            "avg_score": data["total_score"] / data["count"],
        }
        for s, data in all_sentiments.items()
    ]
    sentiment_summary.sort(key=lambda x: x["count"], reverse=True)

    keyword_summary = [
        {
            "keyword": k,
            "count": data["count"],
            "avg_score": data["total_score"] / data["count"],
        }
        for k, data in all_keywords.items()
    ]
    keyword_summary.sort(key=lambda x: x["count"], reverse=True)

    category_summary = [
        {"category": c, "count": count}
        for c, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
    ]

    import statistics

    return {
        "total_documents": len(results),
        "avg_negativity_score": statistics.mean(negativity_scores) if negativity_scores else 0,
        "max_negativity_score": max(negativity_scores) if negativity_scores else 0,
        "min_negativity_score": min(negativity_scores) if negativity_scores else 0,
        "std_negativity_score": statistics.stdev(negativity_scores) if len(negativity_scores) > 1 else 0,
        "top_sentiments": sentiment_summary[:10],
        "top_keywords": keyword_summary[:20],
        "category_distribution": category_summary,
    }


def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    remove_numbers: bool = False,
    min_length: int = 0,
) -> str:
    """
    Preprocess text for analysis.

    Args:
        text: Input text.
        lowercase: Convert to lowercase.
        remove_urls: Remove URLs.
        remove_emails: Remove email addresses.
        remove_mentions: Remove @mentions.
        remove_hashtags: Remove #hashtags.
        remove_numbers: Remove standalone numbers.
        min_length: Minimum word length to keep.

    Returns:
        Preprocessed text.
    """
    if not text:
        return ""

    result = text

    if remove_urls:
        result = re.sub(r"https?://\S+|www\.\S+", "", result)

    if remove_emails:
        result = re.sub(r"\S+@\S+", "", result)

    if remove_mentions:
        result = re.sub(r"@\w+", "", result)

    if remove_hashtags:
        result = re.sub(r"#\w+", "", result)

    if remove_numbers:
        result = re.sub(r"\b\d+\b", "", result)

    if lowercase:
        result = result.lower()

    # Clean up whitespace
    result = re.sub(r"\s+", " ", result).strip()

    if min_length > 0:
        words = result.split()
        words = [w for w in words if len(w) >= min_length]
        result = " ".join(words)

    return result


def chunk_text(
    text: str,
    max_length: int = 512,
    overlap: int = 50,
    split_on: str = "sentence",
) -> List[str]:
    """
    Split long text into chunks for processing.

    Args:
        text: Input text.
        max_length: Maximum characters per chunk.
        overlap: Character overlap between chunks.
        split_on: Split strategy - 'sentence', 'paragraph', or 'word'.

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= max_length:
        return [text] if text else []

    chunks = []

    if split_on == "paragraph":
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

    elif split_on == "sentence":
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent

        if current_chunk:
            chunks.append(current_chunk)

    else:  # word
        words = text.split()
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

    return chunks
