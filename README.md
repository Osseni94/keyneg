<p align="center">
  <img src="assets/Keyneg_logo.png" alt="KeyNeg Logo" width="300">
</p>

<h1 align="center">KeyNeg</h1>

<p align="center">
  <strong>A KeyBERT-style Negative Sentiment and Keyword Extractor for Workforce Intelligence</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/keyneg/"><img src="https://img.shields.io/pypi/v/keyneg.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/keyneg/"><img src="https://img.shields.io/pypi/pyversions/keyneg.svg" alt="Python versions"></a>
  <a href="https://pepy.tech/project/keyneg"><img src="https://static.pepy.tech/badge/keyneg" alt="Downloads"></a>
  <a href="https://github.com/Osseni94/keyneg/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

**Author:** Kaossara Osseni
**Email:** admin@grandnasser.com

KeyNeg extracts negative keywords, frustration indicators, and discontent signals from text. Designed for analyzing employee feedback, forum discussions, customer reviews, and more.

## Installation

```bash
# Install from PyPI
pip install keyneg

# With Streamlit app
pip install keyneg[app]

# Full installation (includes zero-shot classification)
pip install keyneg[all]
```

## Quick Start

```python
from keyneg import KeyNeg

# Initialize (uses all-mpnet-base-v2 by default)
kn = KeyNeg()

# Extract negative sentiments
sentiments = kn.extract_sentiments(
    "I'm frustrated with the constant micromanagement and lack of recognition"
)
print(sentiments)
# [('micromanagement', 0.72), ('frustration', 0.68), ('lack of recognition', 0.65), ...]

# Extract negative keywords
keywords = kn.extract_keywords(
    "The toxic culture and burnout is unbearable"
)
print(keywords)
# [('toxic culture', 0.81), ('burnout', 0.75), ('unbearable', 0.62), ...]

# Full analysis
result = kn.analyze("My manager never listens and I'm thinking of quitting")
print(result)
# {
#     'keywords': [...],
#     'sentiments': [...],
#     'top_sentiment': 'poor leadership',
#     'negativity_score': 0.65,
#     'categories': ['work_environment_culture', 'job_satisfaction']
# }
```

## Features

### Sentiment Extraction

Extract predefined negative sentiment categories:

```python
sentiments = kn.extract_sentiments(
    text,
    top_n=5,           # Number of results
    threshold=0.3,     # Minimum similarity score
    diversity=0.0      # MMR diversity (0-1)
)
```

### Keyword Extraction

Extract negative keywords from both taxonomy and document:

```python
keywords = kn.extract_keywords(
    text,
    top_n=10,
    threshold=0.25,
    keyphrase_ngram_range=(1, 2),
    use_taxonomy=True,
    diversity=0.0
)
```

### Batch Processing

Efficiently process multiple documents:

```python
docs = ["Comment 1...", "Comment 2...", "Comment 3..."]

# Batch analysis
results = kn.analyze_batch(docs, show_progress=True)

# Or individually
keywords_batch = kn.extract_keywords_batch(docs)
sentiments_batch = kn.extract_sentiments_batch(docs)
```

### Special Detectors

**Departure Intent Detection:**
```python
result = kn.detect_departure_intent("I'm updating my resume and interviewing")
# {'detected': True, 'confidence': 0.67, 'signals': ['updating resume', 'interviewing']}
```

**Escalation Risk Detection:**
```python
result = kn.detect_escalation_risk("I'm going to contact my lawyer")
# {'detected': True, 'risk_level': 'high', 'signals': ['contact my lawyer']}
```

**Intensity Analysis:**
```python
result = kn.get_intensity("I'm absolutely furious about this")
# {'level': 3, 'label': 'strong', 'indicators': ['furious']}
```

## Taxonomy Categories

KeyNeg includes a comprehensive taxonomy covering:

- **Work Environment & Culture**: toxic culture, harassment, discrimination, favoritism
- **Management Issues**: micromanagement, poor leadership, lack of direction
- **Recognition & Value**: undervalued, unappreciated, credit stolen
- **Workload & Burnout**: exhaustion, overwhelmed, unrealistic deadlines
- **Compensation**: underpaid, pay disparity, poor benefits
- **Career Development**: no growth, dead end job, glass ceiling
- **Work-Life Balance**: excessive hours, no flexibility
- **Team Dynamics**: conflict, poor collaboration, isolation
- **Job Satisfaction**: low morale, frustration, disengagement
- **Customer/Product Issues**: poor quality, bad service, overpriced

## Customization

### Add Custom Labels

```python
kn.add_custom_labels(["impostor syndrome", "quiet firing"])
```

### Add Custom Keywords

```python
kn.add_custom_keywords("tech_specific", [
    "pager duty", "on-call nightmare", "technical debt"
])
```

### Use Custom Model

```python
kn = KeyNeg(model="all-MiniLM-L6-v2")  # Faster, slightly less accurate
```

## Utility Functions

```python
from keyneg.utils import (
    highlight_keywords,      # Highlight detected keywords in text
    score_to_severity,       # Convert score to severity label
    aggregate_batch_results, # Aggregate batch statistics
    export_to_json,          # Export results to JSON
    export_batch_to_csv,     # Export batch to CSV
    preprocess_text,         # Clean/preprocess text
    chunk_text,              # Split long text into chunks
)

# Highlight keywords in HTML
highlighted = highlight_keywords(text, keywords, format="html")

# Get severity
severity = score_to_severity(0.75)  # "critical"

# Aggregate batch results
summary = aggregate_batch_results(results)
print(summary['top_sentiments'])
print(summary['avg_negativity_score'])
```

## Streamlit App

Launch the interactive UI:

```bash
streamlit run keyneg_app.py
```

Features:
- Single text analysis with detailed results
- Batch processing with file upload
- Interactive visualizations
- Export results to CSV

## Use Cases

1. **Employee Survey Analysis**: Identify patterns of dissatisfaction across responses
2. **Exit Interview Processing**: Extract reasons for departure at scale
3. **Forum Monitoring**: Track sentiment on workforce forums (e.g., TheLayoffradar.com, Blind)
4. **Customer Feedback**: Analyze product reviews and support tickets
5. **Social Media Monitoring**: Track brand sentiment and complaints

## API Integration

```python
from fastapi import FastAPI
from keyneg import KeyNeg

app = FastAPI()
kn = KeyNeg()

@app.post("/analyze")
def analyze(text: str):
    return kn.analyze(text)

@app.post("/analyze_batch")
def analyze_batch(texts: list):
    return kn.analyze_batch(texts)
```

## License

MIT License

## Author

**Kaossara Osseni**
Email: admin@grandnasser.com
GitHub: https://github.com/Osseni94
