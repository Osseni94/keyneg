"""
KeyNeg Package Setup
====================
Install with: pip install -e .

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="keyneg",
    version="1.1.0",
    author="Kaossara Osseni",
    author_email="admin@grandnasser.com",
    description="A KeyBERT-style negative sentiment and keyword extractor for workforce intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://grandnasser.com",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "app": [
            "streamlit>=1.20.0",
            "pandas>=1.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "streamlit>=1.20.0",
            "pandas>=1.3.0",
            "transformers>=4.20.0",  # For zero-shot classification
        ],
    },
    keywords=[
        "nlp",
        "sentiment-analysis",
        "keyword-extraction",
        "workforce-intelligence",
        "text-analysis",
        "negative-sentiment",
        "keybert",
        "sentence-transformers",
    ],
    project_urls={
        "Website": "https://grandnasser.com",
    },
)
