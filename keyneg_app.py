"""
KeyNeg Streamlit Application
============================
Interactive UI for negative sentiment and keyword extraction.

Author: Kaossara Osseni
Email: admin@grandnasser.com

Run with:
    streamlit run keyneg_app.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from keyneg import KeyNeg, get_category_labels
from keyneg.utils import (
    highlight_keywords,
    format_results_table,
    score_to_severity,
    aggregate_batch_results,
)


# ---------------------------
# Caching
# ---------------------------

@st.cache_resource(show_spinner="Loading KeyNeg model...")
def load_keyneg(model_name: str = "all-mpnet-base-v2") -> KeyNeg:
    """Load and cache KeyNeg instance."""
    return KeyNeg(model=model_name)


# ---------------------------
# Main Application
# ---------------------------

def main():
    st.set_page_config(
        page_title="KeyNeg: Negative Sentiment Extractor",
        page_icon="",
        layout="wide",
    )

    st.title("KeyNeg")
    st.subheader("Negative Sentiment & Keyword Extractor for Workforce Intelligence")

    st.markdown(
        """
        Analyze text to extract **negative sentiment themes**, **frustration indicators**,
        and **discontent signals**. Perfect for:
        - Employee survey analysis
        - Forum/review monitoring
        - Exit interview analysis
        - Customer complaint analysis
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    model_name = st.sidebar.selectbox(
        "Embedding Model",
        options=[
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-mpnet-base-v2",
        ],
        index=0,
        help="MPNet models generally perform better but are slower.",
    )

    # Load model
    kn = load_keyneg(model_name)

    # Analysis parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Parameters")

    top_n_sentiments = st.sidebar.slider(
        "Max Sentiments",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of sentiment labels to return.",
    )

    top_n_keywords = st.sidebar.slider(
        "Max Keywords",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of keywords to return.",
    )

    sentiment_threshold = st.sidebar.slider(
        "Sentiment Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum similarity score for sentiments.",
    )

    keyword_threshold = st.sidebar.slider(
        "Keyword Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum similarity score for keywords.",
    )

    diversity = st.sidebar.slider(
        "Result Diversity (MMR)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values = more diverse results (less redundancy).",
    )

    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Analysis", "About"])

    # ---------------------------
    # Tab 1: Single Text Analysis
    # ---------------------------
    with tab1:
        default_text = (
            "I'm completely burned out from the constant micromanagement. "
            "My manager never listens to feedback and takes credit for my work. "
            "The toxic culture here is unbearable and I'm seriously considering quitting. "
            "No work-life balance at all."
        )

        text_input = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            height=150,
            placeholder="Paste employee feedback, review, or forum post here...",
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

        if analyze_btn and text_input.strip():
            with st.spinner("Analyzing..."):
                # Full analysis
                result = kn.analyze(
                    text_input,
                    top_n_keywords=top_n_keywords,
                    top_n_sentiments=top_n_sentiments,
                    keyword_threshold=keyword_threshold,
                    sentiment_threshold=sentiment_threshold,
                )

                # Special detections
                intensity = kn.get_intensity(text_input)
                departure = kn.detect_departure_intent(text_input)
                escalation = kn.detect_escalation_risk(text_input)

            # Results layout
            st.markdown("---")

            # Top metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            with metrics_col1:
                st.metric(
                    "Top Sentiment",
                    result["top_sentiment"] or "None detected",
                )

            with metrics_col2:
                st.metric(
                    "Negativity Score",
                    f"{result['negativity_score']:.2f}",
                    help="Average similarity across detected sentiments (0-1)",
                )

            with metrics_col3:
                severity = score_to_severity(result["negativity_score"])
                st.metric("Severity", severity.upper())

            with metrics_col4:
                st.metric(
                    "Intensity Level",
                    f"{intensity['level']}/4 ({intensity['label']})",
                )

            st.markdown("---")

            # Alert boxes for special detections
            alert_col1, alert_col2 = st.columns(2)

            with alert_col1:
                if departure["detected"]:
                    st.error(
                        f"**Departure Intent Detected** (Confidence: {departure['confidence']:.0%})\n\n"
                        f"Signals: {', '.join(departure['signals'])}"
                    )
                else:
                    st.info("No departure intent detected")

            with alert_col2:
                if escalation["detected"]:
                    color_map = {"high": "error", "medium": "warning", "low": "info"}
                    msg_func = getattr(st, color_map.get(escalation["risk_level"], "info"))
                    msg_func(
                        f"**Escalation Risk: {escalation['risk_level'].upper()}**\n\n"
                        f"Signals: {', '.join(escalation['signals'])}"
                    )
                else:
                    st.info("No escalation risk detected")

            st.markdown("---")

            # Detailed results
            results_col1, results_col2 = st.columns(2)

            with results_col1:
                st.markdown("### Detected Sentiments")
                if result["sentiments"]:
                    sentiment_df = pd.DataFrame(
                        [
                            {
                                "Sentiment": s,
                                "Score": f"{score:.3f}",
                                "Severity": score_to_severity(score),
                            }
                            for s, score in result["sentiments"]
                        ]
                    )
                    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)

                    # Bar chart
                    chart_data = pd.DataFrame(
                        {"Sentiment": [s for s, _ in result["sentiments"]],
                         "Score": [score for _, score in result["sentiments"]]}
                    ).set_index("Sentiment")
                    st.bar_chart(chart_data)
                else:
                    st.info("No sentiments detected above threshold.")

            with results_col2:
                st.markdown("### Detected Keywords")
                if result["keywords"]:
                    keyword_df = pd.DataFrame(
                        [
                            {
                                "Keyword": kw,
                                "Score": f"{score:.3f}",
                                "Severity": score_to_severity(score),
                            }
                            for kw, score in result["keywords"]
                        ]
                    )
                    st.dataframe(keyword_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No keywords detected above threshold.")

            # Categories
            if result["categories"]:
                st.markdown("### Detected Categories")
                st.write(", ".join(result["categories"]))

            # Highlighted text
            st.markdown("### Highlighted Text")
            highlighted = highlight_keywords(
                text_input,
                result["keywords"],
                format="html",
                threshold=keyword_threshold,
            )
            st.markdown(highlighted, unsafe_allow_html=True)

    # ---------------------------
    # Tab 2: Batch Analysis
    # ---------------------------
    with tab2:
        st.markdown("### Batch Analysis")
        st.markdown("Analyze multiple texts at once. Enter one text per line or upload a CSV/TXT file.")

        upload_col, input_col = st.columns([1, 1])

        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload file (CSV or TXT)",
                type=["csv", "txt"],
                help="CSV should have a 'text' column. TXT should have one text per line.",
            )

        texts_to_analyze = []

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "text" in df.columns:
                    texts_to_analyze = df["text"].dropna().tolist()
                else:
                    st.error("CSV must have a 'text' column.")
            else:
                content = uploaded_file.read().decode("utf-8")
                texts_to_analyze = [line.strip() for line in content.split("\n") if line.strip()]

            st.success(f"Loaded {len(texts_to_analyze)} texts from file.")

        with input_col:
            batch_input = st.text_area(
                "Or paste texts here (one per line):",
                height=200,
                placeholder="Text 1\nText 2\nText 3...",
            )

            if batch_input.strip():
                manual_texts = [line.strip() for line in batch_input.split("\n") if line.strip()]
                texts_to_analyze.extend(manual_texts)

        if texts_to_analyze:
            st.info(f"Ready to analyze {len(texts_to_analyze)} texts.")

            if st.button("Run Batch Analysis", type="primary"):
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    results = kn.analyze_batch(
                        texts_to_analyze,
                        top_n_keywords=top_n_keywords,
                        top_n_sentiments=top_n_sentiments,
                        show_progress=True,
                    )

                    # Aggregate statistics
                    summary = aggregate_batch_results(results)

                st.success("Analysis complete!")

                # Summary metrics
                st.markdown("### Summary Statistics")
                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                with sum_col1:
                    st.metric("Documents Analyzed", summary["total_documents"])
                with sum_col2:
                    st.metric("Avg Negativity", f"{summary['avg_negativity_score']:.3f}")
                with sum_col3:
                    st.metric("Max Negativity", f"{summary['max_negativity_score']:.3f}")
                with sum_col4:
                    st.metric("Std Dev", f"{summary['std_negativity_score']:.3f}")

                st.markdown("---")

                # Top sentiments across all docs
                st.markdown("### Most Common Sentiments (Across All Texts)")
                if summary["top_sentiments"]:
                    top_sent_df = pd.DataFrame(summary["top_sentiments"][:10])
                    top_sent_df.columns = ["Sentiment", "Count", "Avg Score"]
                    top_sent_df["Avg Score"] = top_sent_df["Avg Score"].apply(lambda x: f"{x:.3f}")
                    st.dataframe(top_sent_df, use_container_width=True, hide_index=True)

                # Category distribution
                st.markdown("### Category Distribution")
                if summary["category_distribution"]:
                    cat_df = pd.DataFrame(summary["category_distribution"])
                    cat_df.columns = ["Category", "Count"]
                    st.bar_chart(cat_df.set_index("Category"))

                # Detailed results table
                st.markdown("### Detailed Results")
                detail_data = []
                for i, (text, result) in enumerate(zip(texts_to_analyze, results)):
                    detail_data.append({
                        "Index": i,
                        "Text (preview)": text[:100] + "..." if len(text) > 100 else text,
                        "Top Sentiment": result["top_sentiment"] or "-",
                        "Negativity": f"{result['negativity_score']:.3f}",
                        "Severity": score_to_severity(result["negativity_score"]),
                        "Categories": ", ".join(result["categories"][:3]),
                    })

                detail_df = pd.DataFrame(detail_data)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                # Download results
                st.markdown("### Download Results")
                csv_data = detail_df.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    data=csv_data,
                    file_name="keyneg_batch_results.csv",
                    mime="text/csv",
                )

    # ---------------------------
    # Tab 3: About
    # ---------------------------
    with tab3:
        st.markdown(
            """
            ## About KeyNeg

            **Author:** Kaossara Osseni
            **Email:** admin@grandnasser.com
            **Website:** https://grandnasser.com

            KeyNeg is a **KeyBERT-style** library for extracting negative sentiment
            and discontent signals from text. It's designed for:

            - **Workforce Intelligence**: Analyze employee surveys, exit interviews,
              and internal forums to identify issues like burnout, toxic culture,
              and management problems.

            - **Marketing Analysis**: Monitor product reviews, customer feedback,
              and social media for complaints and dissatisfaction.

            ### How It Works

            1. **Embedding**: Text is converted to semantic embeddings using
               Sentence Transformers (MPNet by default).

            2. **Similarity Matching**: Document embeddings are compared against
               pre-defined sentiment labels and a comprehensive keyword taxonomy.

            3. **Scoring**: Results are ranked by cosine similarity and filtered
               by threshold.

            ### Taxonomy Categories
            """
        )

        categories = get_category_labels()
        for cat in categories:
            st.markdown(f"- {cat.replace('_', ' ').title()}")

        st.markdown(
            """
            ### Usage in Code

            ```python
            from keyneg import KeyNeg

            # Initialize
            kn = KeyNeg()

            # Extract sentiments
            sentiments = kn.extract_sentiments("I'm frustrated with the micromanagement")

            # Extract keywords
            keywords = kn.extract_keywords("The toxic culture is unbearable")

            # Full analysis
            result = kn.analyze("My manager never listens")
            print(result['top_sentiment'])
            print(result['negativity_score'])

            # Batch processing
            results = kn.analyze_batch(["Text 1", "Text 2", "Text 3"])

            # Special detectors
            departure = kn.detect_departure_intent("I'm updating my resume")
            escalation = kn.detect_escalation_risk("I'm contacting my lawyer")
            ```

            ### API Integration

            KeyNeg can be easily wrapped in a FastAPI or Flask endpoint for
            production use. The batch methods are optimized for processing
            large volumes of text efficiently.
            """
        )


if __name__ == "__main__":
    main()
