"""
KeyNeg negation detection (v1.3)
================================

Token-level negation scope analysis so detector helpers (departure intent,
escalation risk, intensity) skip matches that fall inside a negation
scope. For example, in "I'm not quitting", the token "quitting" is inside
the scope opened by "not" and is not counted as a departure signal.

Algorithm
---------
Walk tokens left-to-right tracking a monotonic ``negation_count`` and a
``window_active_until`` index:

1. Walls (single-token like ``but``/``however``/``.`` or multi-word like
   "on the other hand") reset both ``count`` and ``active_until``.
2. Multi-word negation phrases (``lack of``, ``failed to``, ``by no means``)
   increment ``count``, advance past the phrase, and conditionally open
   the window from the phrase end.
3. Single-token negators (``not``, ``no``, ``never``, ``without``, plus
   verbal negators ``fail``/``refuse``/``deny``/...) increment ``count``
   and conditionally open the window.
4. Double-negation cancellation: if ``count`` is even after the increment,
   the window does NOT open (it cancels). Examples:
   ``not never happy`` (two negators) → ``happy`` not negated.
   ``no objection to escalating`` (negator + objection-as-prefix) →
   ``escalating`` not negated.
5. Negative-prefix awareness: when a negator is followed by a word with
   an inherent negative prefix (``unhappy``, ``incompetent``, ``dislike``),
   ``count`` is incremented by an extra +1, triggering cancellation.

Algorithm ported from ONES-rs/expander.rs (with curated prefix-base
whitelist also from ONES-rs/constants.rs to avoid ``into → to`` false
positives).

Limitations
-----------
- Window-based, not parser-based: long-range negation across multiple
  clauses ("There are no... [much later] ...plans") will be missed.
- Multi-word keywords are checked at their starting token only; a match
  that *starts* inside a negation scope is treated as negated.
- Triple-negation collapses correctly via the modulo, but very rare
  patterns ("not no one") may give surprising results.
"""

import re
from typing import Iterable, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Single-token negators (extended with verbal negators from ONES-rs)
# -----------------------------------------------------------------------------

NEGATION_TOKENS = frozenset({
    # Standard negators
    "not", "no", "never", "none", "nor", "nothing", "nowhere", "neither",
    "nobody", "noone", "nope", "nah", "nay",
    # Adverbs
    "rarely", "seldom", "hardly", "scarcely", "barely",
    "without",
    # Lack family
    "lack", "lacking", "lacks", "lacked",
    # Verbal negators (action-based negation; following object is negated)
    "fail", "fails", "failed", "failing",
    "unable",
    "refuse", "refuses", "refused", "refusing",
    "prevent", "prevents", "prevented", "preventing",
    "deny", "denies", "denied", "denying",
    "reject", "rejects", "rejected", "rejecting",
    "miss", "misses", "missed", "missing",
    "lose", "loses", "lost", "losing",
    "stop", "stops", "stopped", "stopping",
    "block", "blocks", "blocked", "blocking",
    "avoid", "avoids", "avoided", "avoiding",
    "exclude", "excludes", "excluded", "excluding",
    # Without-apostrophe contractions (in case preprocessing strips them)
    "cannot", "wont", "cant", "dont", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "mustnt", "neednt",
})


# -----------------------------------------------------------------------------
# Multi-word negation phrases — checked first (longer matches win)
# -----------------------------------------------------------------------------

# Order: longest phrases first so we don't shadow specific cases with
# generic prefixes (e.g. "no chance of" must be checked before "no").
NEGATION_PHRASES: Tuple[str, ...] = (
    "out of the question",
    "under no circumstances",
    "fall short of", "falls short of", "fell short of", "falling short of",
    "no chance of", "no intention of",
    "no plans to", "no plans for", "no plans on",
    "no need to", "no need for",
    "by no means", "in no way", "on no account",
    "have no", "has no", "had no", "having no",
    "without any",
    "failed to", "fails to", "fail to", "failing to",
    "refused to", "refuses to", "refuse to", "refusing to",
    "unable to",
    "no longer",
    "lack of", "lacks of",
    "absence of",
    "instead of",
    "rather than",
    "far from",
    "anything but",
    "not really",
    "not at all",
)


# -----------------------------------------------------------------------------
# Walls — close the negation scope (single-token + multi-word)
# -----------------------------------------------------------------------------

CLAUSE_BOUNDARIES = frozenset({
    # Punctuation
    ".", ";", ":", "!", "?", ",",
    # Single-token discourse markers
    "but", "however", "although", "though", "yet",
    "nevertheless", "nonetheless", "whereas", "while", "except",
    "instead", "notwithstanding", "conversely", "alternatively",
    "otherwise", "contrarily", "regardless", "still",
})

MULTI_WORD_WALLS: Tuple[str, ...] = (
    "on the other hand",
    "in contrast",
    "on the contrary",
    "in spite of",
    "despite the fact",
    "even though",
    "having said that",
    "that being said",
    "that said",
)


# -----------------------------------------------------------------------------
# Negative-prefix words (curated whitelist, ported from ONES-rs)
# Avoids false positives like "into → to" or "intention → tention".
# -----------------------------------------------------------------------------

NEGATIVE_PREFIX_WORDS = frozenset({
    # un-
    "unhappy", "unsafe", "unreliable", "unable", "uncertain", "unclear",
    "uncomfortable", "uncommon", "unconscious", "undecided", "undefined",
    "unequal", "unexpected", "unfair", "unfamiliar", "unfortunate",
    "unfriendly", "unhealthy", "unhelpful", "unimportant", "uninterested",
    "unkind", "unknown", "unlikely", "unlimited", "unlucky", "unnatural",
    "unnecessary", "unpleasant", "unpopular", "unprepared", "unproductive",
    "unprofessional", "unreal", "unreasonable", "unsatisfied", "unstable",
    "unsuccessful", "unsure", "untrue", "unusual", "unwilling", "unwise",
    # in-
    "incompetent", "incomplete", "inconsistent", "incorrect", "indirect",
    "ineffective", "inefficient", "inexperienced", "informal", "infrequent",
    "insecure", "insensitive", "insignificant", "insincere", "insufficient",
    "invalid", "invisible",
    # im-
    "imbalanced", "immature", "immoral", "impatient", "imperfect",
    "impossible", "improper", "impure",
    # dis-
    "disagree", "disappear", "disapprove", "dishonest", "dislike",
    "disloyal", "disobey", "disorder", "disorganized", "displease",
    "disrespect", "dissatisfied", "distrust",
    # ir-
    "irrational", "irregular", "irrelevant", "irresponsible", "irreversible",
    # il-
    "illegal", "illegible", "illogical", "illegitimate", "illiterate",
})


DEFAULT_WINDOW = 4

_TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b|[.;:!?,]", re.UNICODE)


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def has_negative_prefix(word: str) -> bool:
    """True if ``word`` is in the curated set of inherently-negated words."""
    return word.lower() in NEGATIVE_PREFIX_WORDS


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize ``text`` into ``(lower_token, start_char, end_char)`` triples.

    Contractions ending in ``n't`` (don't, can't, won't, ...) are split into
    their stem plus an explicit ``"not"`` token so negation propagates.
    """
    tokens: List[Tuple[str, int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0)
        start, end = match.start(), match.end()
        lower = tok.lower()
        if lower.endswith("n't") and len(lower) > 3:
            split = end - 3
            tokens.append((lower[:-3], start, split))
            tokens.append(("not", split, end))
        else:
            tokens.append((lower, start, end))
    return tokens


# -----------------------------------------------------------------------------
# Phrase matching
# -----------------------------------------------------------------------------

def _starts_with_phrase(tokens: List[Tuple[str, int, int]],
                        start: int,
                        phrase: str) -> int:
    """If ``tokens[start:]`` starts with ``phrase``, return its token length.

    Returns 0 on mismatch. Phrase is split on whitespace and compared
    token-by-lowercased-token, so this is robust to spacing/punctuation
    variance in the input text.
    """
    parts = phrase.split()
    if start + len(parts) > len(tokens):
        return 0
    for offset, part in enumerate(parts):
        if tokens[start + offset][0] != part:
            return 0
    return len(parts)


def _check_multi_word_wall(tokens: List[Tuple[str, int, int]],
                           start: int) -> int:
    for phrase in MULTI_WORD_WALLS:
        length = _starts_with_phrase(tokens, start, phrase)
        if length:
            return length
    return 0


def _check_multi_word_negation(tokens: List[Tuple[str, int, int]],
                               start: int) -> int:
    for phrase in NEGATION_PHRASES:
        length = _starts_with_phrase(tokens, start, phrase)
        if length:
            return length
    return 0


# -----------------------------------------------------------------------------
# Core negation scope detection
# -----------------------------------------------------------------------------

def negated_indices(tokens: List[Tuple[str, int, int]],
                    window: int = DEFAULT_WINDOW,
                    extra_negation_tokens: Optional[Iterable[str]] = None) -> Set[int]:
    """Return the set of token indices that fall inside an active negation scope.

    Args:
        tokens: Output of ``tokenize``.
        window: How many tokens after a negator are inside its scope. The
            default of 4 follows VADER / NegEx convention.
        extra_negation_tokens: Optional iterable of additional negator
            cues to recognize on top of ``NEGATION_TOKENS``. Lower-cased
            internally. Useful for domain idioms like ``notwithstanding``.

    Returns:
        Set of token indices marked as inside a negation scope. Negators
        themselves are NOT in this set; only the *content tokens* they
        scope over.
    """
    extra: Set[str] = set()
    if extra_negation_tokens:
        extra = {t.lower() for t in extra_negation_tokens if t}

    negated: Set[int] = set()
    count = 0
    active_until = -1
    skip_until = 0

    i = 0
    n = len(tokens)
    while i < n:
        if i < skip_until:
            i += 1
            continue

        tok = tokens[i][0]

        # ------- User-supplied negators take precedence over walls/phrases.
        # (A user adding 'notwithstanding' to extra_negation_tokens wants
        # it treated as a negator even though it's also a default wall.)
        if tok in extra:
            count += 1
            if i + 1 < n and has_negative_prefix(tokens[i + 1][0]):
                count += 1
            if count % 2 == 1:
                active_until = i + window
            else:
                active_until = -1
            i += 1
            continue

        # ------- Walls: reset count and window -------
        if tok in CLAUSE_BOUNDARIES:
            count = 0
            active_until = -1
            i += 1
            continue

        wall_len = _check_multi_word_wall(tokens, i)
        if wall_len:
            count = 0
            active_until = -1
            skip_until = i + wall_len
            i += 1
            continue

        # ------- Multi-word negation phrase -------
        phrase_len = _check_multi_word_negation(tokens, i)
        if phrase_len:
            count += 1
            after = i + phrase_len
            # "lack of unfair X" — token after the phrase is itself
            # inherently negated; counts as a second negation.
            if after < n and has_negative_prefix(tokens[after][0]):
                count += 1
            skip_until = after
            if count % 2 == 1:
                # window covers `window` tokens starting at index `after`.
                active_until = after + window - 1
            else:
                active_until = -1
            i += 1
            continue

        # ------- Single-token negator -------
        if tok in NEGATION_TOKENS:
            count += 1
            # "not unhappy" — next token has a negative prefix → cancels.
            if i + 1 < n and has_negative_prefix(tokens[i + 1][0]):
                count += 1
            if count % 2 == 1:
                active_until = i + window
            else:
                active_until = -1
            i += 1
            continue

        # ------- Regular content token -------
        if i <= active_until:
            negated.add(i)

        i += 1

    return negated


# -----------------------------------------------------------------------------
# Public API: keyword filtering
# -----------------------------------------------------------------------------

def _word_boundary_pattern(keyword: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(keyword.lower().strip()) + r"\b", re.IGNORECASE)


def find_unnegated_matches(doc: str,
                           keywords: Iterable[str],
                           window: int = DEFAULT_WINDOW,
                           extra_negation_tokens: Optional[Iterable[str]] = None
                           ) -> List[str]:
    """Return ``keywords`` that occur at least once *outside* a negation scope.

    A keyword that appears only in negated form is dropped. Returned matches
    preserve the original casing supplied in ``keywords``.

    Args:
        doc: Text to search.
        keywords: Iterable of keyword strings to look up.
        window: Negation window size.
        extra_negation_tokens: Domain-specific negators to add on top of
            the built-in set (forwarded to ``negated_indices``).
    """
    if not doc:
        return []

    tokens = tokenize(doc)
    if not tokens:
        return []

    negated = negated_indices(
        tokens, window=window, extra_negation_tokens=extra_negation_tokens
    )

    if not negated:
        # Fast path: no negation scopes → just check substring presence.
        doc_lower = doc.lower()
        return [
            kw for kw in keywords
            if kw and kw.strip() and _word_boundary_pattern(kw).search(doc_lower)
        ]

    char_to_tok = {}
    for idx, (_tok, start, end) in enumerate(tokens):
        for c in range(start, end):
            char_to_tok[c] = idx

    doc_lower = doc.lower()
    matched: List[str] = []
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        pattern = _word_boundary_pattern(kw)
        for match in pattern.finditer(doc_lower):
            tok_idx = char_to_tok.get(match.start())
            if tok_idx is None:
                continue
            if tok_idx not in negated:
                matched.append(kw)
                break
    return matched


def is_negated(doc: str, keyword: str,
               window: int = DEFAULT_WINDOW,
               extra_negation_tokens: Optional[Iterable[str]] = None) -> bool:
    """Return True iff ``keyword`` occurs in ``doc`` and every occurrence is negated."""
    if not doc or not keyword or not keyword.strip():
        return False

    pattern = _word_boundary_pattern(keyword)
    if not pattern.search(doc.lower()):
        return False

    matches = find_unnegated_matches(
        doc, [keyword], window=window,
        extra_negation_tokens=extra_negation_tokens,
    )
    return not bool(matches)
