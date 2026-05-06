"""Lightweight packaging-consistency tests."""

from pathlib import Path
import re

import keyneg


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_string_format():
    assert re.match(r"^\d+\.\d+\.\d+", keyneg.__version__)


def test_version_consistent_with_pyproject():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "version not found in pyproject.toml"
    assert match.group(1) == keyneg.__version__


def test_taxonomy_module_version_consistent():
    tax_path = REPO_ROOT / "keyneg" / "taxonomy.py"
    text = tax_path.read_text(encoding="utf-8")
    # The docstring used to disagree with __init__.py — pin them together.
    match = re.search(r"Version:\s*([0-9]+\.[0-9]+\.[0-9]+)", text)
    if match:
        assert match.group(1) == keyneg.__version__


def test_no_top_level_setup_py():
    # We removed setup.py in favor of a single pyproject.toml source of truth.
    assert not (REPO_ROOT / "setup.py").exists()


def test_no_top_level_keyneg_app_py():
    # The Streamlit app moved into the package as keyneg/app.py.
    assert not (REPO_ROOT / "keyneg_app.py").exists()


def test_app_runner_importable():
    from keyneg import _app_runner
    assert callable(_app_runner.main)


def test_public_api_exports_complete():
    expected = {
        "KeyNeg",
        "SENTIMENT_LABELS",
        "NEGATIVE_TAXONOMY",
        "get_all_keywords",
        "get_keywords_by_category",
        "get_category_labels",
        "find_unnegated_matches",
        "is_negated",
        "NEGATION_TOKENS",
        "PolarityClassifier",
        "PolarityError",
        "get_polarity_classifier",
    }
    assert expected <= set(keyneg.__all__)
