"""Tests for the negation-aware detector helpers.

The original community/enterprise code used naive substring matching, so
"I'm not quitting" would still trigger ``detect_departure_intent``. This
suite locks in the corrected behavior.
"""

from keyneg.negation import (
    NEGATION_TOKENS,
    find_unnegated_matches,
    is_negated,
    negated_indices,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer behavior
# ---------------------------------------------------------------------------

def test_tokenize_splits_nt_contractions():
    tokens = tokenize("I don't quit.")
    surface = [t for t, _, _ in tokens]
    # don't → "do" + "not" (we strip the trailing "n't" suffix).
    assert "not" in surface
    assert "do" in surface
    assert surface.index("not") == surface.index("do") + 1


def test_tokenize_keeps_clause_punctuation_as_tokens():
    tokens = tokenize("I am not quitting. I love this place.")
    surface = [t for t, _, _ in tokens]
    assert "." in surface
    # The period appears between the two clauses.
    assert surface.count(".") == 2


def test_tokenize_handles_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# Negation scope detection
# ---------------------------------------------------------------------------

def test_negation_scope_basic():
    tokens = tokenize("I am not quitting")
    negated = negated_indices(tokens)
    quit_idx = [i for i, (t, _, _) in enumerate(tokens) if t == "quitting"][0]
    assert quit_idx in negated


def test_negation_scope_resets_at_clause_boundary():
    tokens = tokenize("I am not quitting. I am leaving.")
    negated = negated_indices(tokens)
    leaving_idx = [i for i, (t, _, _) in enumerate(tokens) if t == "leaving"][0]
    assert leaving_idx not in negated, "Negation must reset at sentence boundary"


def test_negation_scope_resets_at_contrastive_conjunction():
    tokens = tokenize("I am not quitting but I am leaving")
    negated = negated_indices(tokens)
    leaving_idx = [i for i, (t, _, _) in enumerate(tokens) if t == "leaving"][0]
    assert leaving_idx not in negated, "'but' must close negation scope"


def test_negation_window_default():
    # window=4 means the 4 tokens after a negator are negated.
    tokens = tokenize("not a b c d e")
    negated = negated_indices(tokens, window=4)
    # tokens: not(0) a(1) b(2) c(3) d(4) e(5) → 1..4 negated, 5 free
    surface = [t for t, _, _ in tokens]
    e_idx = surface.index("e")
    assert e_idx not in negated


def test_no_negators_means_no_negation():
    tokens = tokenize("I am quitting and I will leave.")
    negated = negated_indices(tokens)
    assert negated == set()


# ---------------------------------------------------------------------------
# find_unnegated_matches — the public API
# ---------------------------------------------------------------------------

def test_quitting_negated_is_filtered_out():
    matches = find_unnegated_matches("I'm not quitting", ["quitting"])
    assert matches == []


def test_quitting_unnegated_is_kept():
    matches = find_unnegated_matches("I'm quitting", ["quitting"])
    assert matches == ["quitting"]


def test_no_plans_to_leave_is_filtered():
    matches = find_unnegated_matches("I have no plans to leave", ["leave"])
    assert matches == []


def test_without_complaints_is_filtered():
    matches = find_unnegated_matches("Without any complaints from staff", ["complaints"])
    assert matches == []


def test_never_escalating_is_filtered():
    matches = find_unnegated_matches(
        "I am never escalating this to legal",
        ["escalating", "legal"],
    )
    assert matches == []


def test_dont_contraction_negation():
    matches = find_unnegated_matches("I don't intend to quit", ["quit"])
    assert matches == []


def test_double_clause_one_negated_one_not():
    matches = find_unnegated_matches(
        "I'm not quitting. But I am updating my resume.",
        ["quitting", "updating my resume"],
    )
    # 'quitting' is negated, but 'updating my resume' is not.
    assert matches == ["updating my resume"]


def test_multiple_occurrences_one_unnegated_keeps_keyword():
    # If a keyword appears in BOTH negated and unnegated form, keep it.
    matches = find_unnegated_matches(
        "I am not quitting today. I am quitting tomorrow.",
        ["quitting"],
    )
    assert matches == ["quitting"]


def test_multiword_keyword_starts_in_negation_scope():
    matches = find_unnegated_matches(
        "I'm not contacting my lawyer about this",
        ["contacting my lawyer", "lawyer"],
    )
    assert matches == []


def test_empty_inputs_handled():
    assert find_unnegated_matches("", ["quit"]) == []
    assert find_unnegated_matches("doc", []) == []
    assert find_unnegated_matches("doc", [""]) == []


def test_unrelated_text_no_keywords():
    matches = find_unnegated_matches("The weather is nice today.", ["quit", "leave"])
    assert matches == []


def test_word_boundary_prevents_substring_match():
    # 'quit' must not match inside 'quite'.
    matches = find_unnegated_matches("I'm quite happy with the team.", ["quit"])
    assert matches == []


def test_case_insensitive_match():
    matches = find_unnegated_matches("I'm QUITTING tomorrow", ["quitting"])
    assert matches == ["quitting"]


def test_is_negated_helper():
    assert is_negated("I'm not quitting", "quitting") is True
    assert is_negated("I'm quitting", "quitting") is False
    # Keyword absent entirely → False per the contract.
    assert is_negated("hello world", "quitting") is False


def test_negation_tokens_include_common_negators():
    for cue in {"not", "no", "never", "without", "neither", "lacks"}:
        assert cue in NEGATION_TOKENS
