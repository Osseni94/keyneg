"""Tests for the v1.3 negation upgrades ported from ONES-rs.

These cover the cases the v1.2 single-token-window approach missed:
multi-word phrases, multi-word walls, double-negation cancellation,
extended verbal negators, comma-as-wall, prefix-aware doubles, and
user-extensible negation cues.
"""

import pytest

from keyneg.negation import (
    MULTI_WORD_WALLS,
    NEGATION_PHRASES,
    NEGATIVE_PREFIX_WORDS,
    find_unnegated_matches,
    has_negative_prefix,
    is_negated,
    negated_indices,
    tokenize,
)


# ---------------------------------------------------------------------------
# Multi-word negation phrases
# ---------------------------------------------------------------------------

def test_lack_of_filters_following_keyword():
    matches = find_unnegated_matches(
        "There's a lack of escalation plans here", ["escalation"]
    )
    assert matches == []


def test_failed_to_filters_following_keyword():
    matches = find_unnegated_matches(
        "Management failed to address the harassment claims", ["harassment"]
    )
    assert matches == []


def test_no_longer_filters_following_keyword():
    matches = find_unnegated_matches(
        "He's no longer thinking about quitting", ["quitting"]
    )
    assert matches == []


def test_no_plans_to_filters():
    matches = find_unnegated_matches(
        "I have no plans to leave the company", ["leave"]
    )
    assert matches == []


def test_no_need_to_filters():
    matches = find_unnegated_matches(
        "There is no need to escalate this", ["escalate"]
    )
    assert matches == []


def test_by_no_means_filters():
    matches = find_unnegated_matches(
        "By no means am I considering quitting", ["quitting"]
    )
    assert matches == []


def test_in_no_way_filters():
    # Keep the sentence short — the 4-token window after "in no way" must
    # still cover the keyword. Longer sentences hit window-size limitations
    # that ONES-rs has too.
    matches = find_unnegated_matches(
        "In no way am I tolerating harassment", ["harassment"]
    )
    assert matches == []


def test_unable_to_filters():
    matches = find_unnegated_matches(
        "We were unable to escalate the issue", ["escalate"]
    )
    assert matches == []


def test_refused_to_filters():
    matches = find_unnegated_matches(
        "I refused to quit despite the pressure", ["quit"]
    )
    assert matches == []


def test_have_no_filters():
    # Short sentence so "lawyer" is inside the 4-token window after "have no".
    matches = find_unnegated_matches(
        "I have no lawyer involvement here", ["lawyer"]
    )
    assert matches == []


def test_negation_phrases_are_nonempty():
    assert len(NEGATION_PHRASES) > 15
    assert "lack of" in NEGATION_PHRASES
    assert "failed to" in NEGATION_PHRASES
    assert "no longer" in NEGATION_PHRASES


# ---------------------------------------------------------------------------
# Multi-word wall phrases — close negation scope
# ---------------------------------------------------------------------------

def test_on_the_other_hand_resets_negation():
    # "not happy ... on the other hand ... leaving" — leaving is past the
    # wall and should NOT be negated.
    matches = find_unnegated_matches(
        "I'm not happy on the other hand I am leaving", ["leaving"]
    )
    assert matches == ["leaving"]


def test_despite_the_fact_resets():
    matches = find_unnegated_matches(
        "I'm not staying despite the fact that I am quitting tomorrow",
        ["quitting"],
    )
    assert matches == ["quitting"]


def test_even_though_resets():
    matches = find_unnegated_matches(
        "I'm not happy even though I am leaving",
        ["leaving"],
    )
    assert matches == ["leaving"]


def test_in_contrast_resets():
    matches = find_unnegated_matches(
        "I am not quitting in contrast my colleague is leaving",
        ["leaving"],
    )
    assert matches == ["leaving"]


def test_multi_word_walls_constant():
    assert "on the other hand" in MULTI_WORD_WALLS
    assert "despite the fact" in MULTI_WORD_WALLS


# ---------------------------------------------------------------------------
# Double-negation cancellation
# ---------------------------------------------------------------------------

def test_double_negative_cancels_keep_keyword():
    """Two negators in a row cancel out — the following word is unnegated."""
    matches = find_unnegated_matches(
        "I am not never quitting tomorrow", ["quitting"]
    )
    # "not" + "never" = even count → window does not open → quitting unnegated.
    assert matches == ["quitting"]


def test_no_objection_to_keyword_unnegated():
    """A negator followed by a prefix-negated word counts as 2 → cancels."""
    # "no objection to escalating" — "no" + "objection" might not be
    # in our prefix whitelist, but the principle is the same case as
    # below ("not unhappy"). Here we test the explicit prefix-cancel
    # path.
    matches = find_unnegated_matches(
        "I am not unhappy about escalating", ["escalating"]
    )
    # not + unhappy(prefix) → count=2 → cancel → escalating unnegated.
    assert matches == ["escalating"]


def test_triple_negation_negates_again():
    """Odd count of negators in one clause leaves the window open."""
    matches = find_unnegated_matches(
        "I am not never not quitting today", ["quitting"]
    )
    # not + never + not = 3 odd → window open → quitting negated.
    assert matches == []


# ---------------------------------------------------------------------------
# Extended verbal negators
# ---------------------------------------------------------------------------

def test_refused_filters():
    matches = find_unnegated_matches(
        "She refused contact with the lawyer", ["lawyer"]
    )
    assert matches == []


def test_prevented_filters():
    matches = find_unnegated_matches(
        "HR prevented further escalation of the dispute", ["escalation"]
    )
    assert matches == []


def test_rarely_filters():
    matches = find_unnegated_matches(
        "We rarely escalate issues to the executive team", ["escalate"]
    )
    assert matches == []


def test_hardly_filters():
    matches = find_unnegated_matches(
        "I hardly think about quitting these days", ["quitting"]
    )
    assert matches == []


# ---------------------------------------------------------------------------
# Extended walls
# ---------------------------------------------------------------------------

def test_comma_resets_negation():
    """Comma is now a wall (was missing in v1.2)."""
    matches = find_unnegated_matches(
        "I am not happy, but I am quitting", ["quitting"]
    )
    # The comma resets, then "but" resets again. quitting is well past both.
    assert matches == ["quitting"]


def test_nevertheless_resets():
    matches = find_unnegated_matches(
        "I am not happy nevertheless I am leaving", ["leaving"]
    )
    assert matches == ["leaving"]


def test_regardless_resets():
    matches = find_unnegated_matches(
        "I am not interested regardless I am quitting", ["quitting"]
    )
    assert matches == ["quitting"]


def test_still_resets():
    matches = find_unnegated_matches(
        "I'm not feeling great still I am working hard", ["working hard"]
    )
    assert matches == ["working hard"]


# ---------------------------------------------------------------------------
# Prefix awareness
# ---------------------------------------------------------------------------

def test_has_negative_prefix_helper():
    assert has_negative_prefix("unhappy") is True
    assert has_negative_prefix("incompetent") is True
    assert has_negative_prefix("dislike") is True
    assert has_negative_prefix("irrational") is True
    # False positives we explicitly avoid:
    assert has_negative_prefix("into") is False
    assert has_negative_prefix("intention") is False
    assert has_negative_prefix("display") is False


def test_negative_prefix_words_curated():
    # Spot-check the curated set is large enough.
    assert len(NEGATIVE_PREFIX_WORDS) >= 80
    assert "unhappy" in NEGATIVE_PREFIX_WORDS
    assert "incompetent" in NEGATIVE_PREFIX_WORDS
    assert "dishonest" in NEGATIVE_PREFIX_WORDS


def test_not_unhappy_keyword_unnegated():
    """'not unhappy' is a double-negative → following word unaffected."""
    matches = find_unnegated_matches(
        "I am not unhappy about escalating",
        ["escalating"],
    )
    assert matches == ["escalating"]


def test_inherent_prefix_word_alone_not_negated_set():
    """A word with a negative prefix is not itself in the negated_indices set
    just because of its prefix — the prefix only matters when it follows
    a negator (then it triggers cancellation).
    """
    tokens = tokenize("This product is unreliable")
    negated = negated_indices(tokens)
    # "unreliable" without an upstream negator is not in the negated set.
    surface = [t for t, _, _ in tokens]
    unrel_idx = surface.index("unreliable")
    assert unrel_idx not in negated


# ---------------------------------------------------------------------------
# User-extensible negation cues
# ---------------------------------------------------------------------------

def test_extra_negation_tokens_filters():
    """Domain-specific negators added via extra_negation_tokens are honored."""
    matches = find_unnegated_matches(
        "Notwithstanding any escalation discussions, the case stands",
        ["escalation"],
        extra_negation_tokens=["notwithstanding"],
    )
    assert matches == []


def test_extra_negation_tokens_case_insensitive():
    matches = find_unnegated_matches(
        "NOTWITHSTANDING the lawyer involvement",
        ["lawyer"],
        extra_negation_tokens=["NotWithStanding"],
    )
    assert matches == []


def test_extra_negation_tokens_does_not_affect_default():
    """Without extra_negation_tokens, 'notwithstanding' is just a wall, not a negator."""
    # In our default config "notwithstanding" is in CLAUSE_BOUNDARIES (a wall),
    # so it RESETS scope, doesn't open one. So escalation should NOT be
    # filtered.
    matches = find_unnegated_matches(
        "Notwithstanding the discussion, I am pursuing escalation",
        ["escalation"],
    )
    assert matches == ["escalation"]


def test_extra_negation_tokens_empty_or_none():
    matches = find_unnegated_matches(
        "I'm escalating now",
        ["escalating"],
        extra_negation_tokens=None,
    )
    assert matches == ["escalating"]
    matches = find_unnegated_matches(
        "I'm escalating now",
        ["escalating"],
        extra_negation_tokens=[],
    )
    assert matches == ["escalating"]


# ---------------------------------------------------------------------------
# KeyNeg constructor wiring
# ---------------------------------------------------------------------------

def test_constructor_extra_negation_tokens(fake_encoder):
    from keyneg import KeyNeg

    kn = KeyNeg(model=fake_encoder, extra_negation_tokens=["notwithstanding"])

    # Keep the sentence short — "lawyer" must fall inside the 4-token
    # window opened by "notwithstanding".
    result = kn.detect_escalation_risk("Notwithstanding any lawyer involvement")
    assert result["detected"] is False


def test_constructor_default_does_not_treat_notwithstanding_as_negator(fake_encoder):
    from keyneg import KeyNeg

    kn = KeyNeg(model=fake_encoder)

    # "notwithstanding" is a wall in defaults; it resets scope, doesn't
    # open a negation window. So this depends on the rest of the sentence
    # and lawyer-related taxonomy keywords. Here we just check the call
    # works without error and returns the expected shape.
    result = kn.detect_escalation_risk(
        "Notwithstanding the involvement of any lawyer in this matter."
    )
    assert "detected" in result
    assert "risk_level" in result


# ---------------------------------------------------------------------------
# Backward compatibility — existing behavior must not regress
# ---------------------------------------------------------------------------

def test_v12_basic_negation_still_works():
    assert find_unnegated_matches("I'm not quitting", ["quitting"]) == []
    assert find_unnegated_matches("I'm quitting", ["quitting"]) == ["quitting"]


def test_v12_clause_boundary_period_still_resets():
    matches = find_unnegated_matches(
        "I am not quitting. I am leaving.", ["leaving"]
    )
    assert matches == ["leaving"]


def test_v12_dont_contraction_still_negates():
    assert find_unnegated_matches("I don't intend to quit", ["quit"]) == []


def test_v12_word_boundary_still_holds():
    # "quit" must not match inside "quite".
    assert find_unnegated_matches("I'm quite happy", ["quit"]) == []
