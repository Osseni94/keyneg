"""
KeyNeg Test - General Motors CSV Analysis
==========================================
Analyze GM posts from CSV file.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import sys
from pathlib import Path
import pandas as pd
import re

# Add keyneg to path
sys.path.insert(0, str(Path(__file__).parent))

from keyneg import KeyNeg
from keyneg.utils import aggregate_batch_results, score_to_severity, export_batch_to_csv

def main():
    csv_path = r"C:\Users\User\The Prof\thelayoff_scraper\General_Motors_Week0_Raw_files.csv"

    print("=" * 80)
    print("KeyNeg Analysis - General Motors Posts")
    print("=" * 80)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Get the posts column
    content_col = 'Posts' if 'Posts' in df.columns else df.columns[0]

    # The posts are combined with newlines - split them into individual posts
    all_posts = []
    for _, row in df.iterrows():
        content = row[content_col]
        if pd.notna(content):
            # Split by newlines and filter empty lines
            posts = [p.strip() for p in str(content).split('\n') if p.strip()]
            all_posts.extend(posts)

    # Filter out very short posts (likely not meaningful)
    texts = [p for p in all_posts if len(p) > 20]

    print(f"Total individual posts extracted: {len(texts)}")

    # Show sample posts
    print("\n" + "-" * 80)
    print("SAMPLE POSTS:")
    print("-" * 80)
    for i, text in enumerate(texts[:5]):
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"\n[Post {i+1}]: {preview}")

    # Initialize KeyNeg
    print("\n" + "=" * 80)
    print("Initializing KeyNeg...")
    print("=" * 80)

    kn = KeyNeg(model="all-mpnet-base-v2")
    print(f"Model loaded: {kn}")

    # Analyze all posts
    print(f"\nAnalyzing {len(texts)} posts...")
    results = kn.analyze_batch(texts, show_progress=True)

    # Aggregate
    summary = aggregate_batch_results(results)

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\nDocuments analyzed: {summary['total_documents']}")
    print(f"Average negativity score: {summary['avg_negativity_score']:.3f}")
    print(f"Max negativity score: {summary['max_negativity_score']:.3f}")
    print(f"Min negativity score: {summary['min_negativity_score']:.3f}")
    print(f"Std deviation: {summary['std_negativity_score']:.3f}")

    # Top sentiments
    print("\n" + "-" * 80)
    print("TOP 15 SENTIMENTS (Most Frequent)")
    print("-" * 80)
    for item in summary['top_sentiments'][:15]:
        bar = "#" * int(item['avg_score'] * 30)
        print(f"  {item['sentiment']:<35} Count: {item['count']:>4}  Avg: {item['avg_score']:.3f} {bar}")

    # Top keywords - THIS IS THE MAIN OUTPUT
    print("\n" + "-" * 80)
    print("TOP 30 NEGATIVE KEYWORDS EXTRACTED")
    print("-" * 80)
    for i, item in enumerate(summary['top_keywords'][:30], 1):
        bar = "#" * int(item['avg_score'] * 30)
        print(f"  {i:>2}. {item['keyword']:<35} Count: {item['count']:>4}  Avg: {item['avg_score']:.3f} {bar}")

    # Category distribution
    print("\n" + "-" * 80)
    print("CATEGORY DISTRIBUTION")
    print("-" * 80)
    for item in summary['category_distribution']:
        pct = 100 * item['count'] / len(texts)
        bar = "#" * int(pct)
        print(f"  {item['category']:<40} {item['count']:>4} ({pct:5.1f}%) {bar}")

    # Most negative posts
    print("\n" + "-" * 80)
    print("TOP 10 MOST NEGATIVE POSTS")
    print("-" * 80)

    scored_posts = list(zip(range(len(texts)), texts, results))
    scored_posts.sort(key=lambda x: x[2]['negativity_score'], reverse=True)

    for idx, text, result in scored_posts[:10]:
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n[Post {idx+1}] Negativity: {result['negativity_score']:.3f} ({score_to_severity(result['negativity_score']).upper()})")
        print(f"  Top Sentiment: {result['top_sentiment']}")
        print(f"  Keywords: {', '.join([k[0] for k in result['keywords'][:5]])}")
        print(f"  Text: {preview}")

    # Departure intent
    print("\n" + "-" * 80)
    print("DEPARTURE INTENT DETECTION")
    print("-" * 80)

    departure_posts = []
    for idx, text in enumerate(texts):
        departure = kn.detect_departure_intent(text)
        if departure['detected']:
            departure_posts.append((idx, text, departure))

    print(f"Posts with departure intent: {len(departure_posts)} / {len(texts)} ({100*len(departure_posts)/len(texts):.1f}%)")

    for idx, text, dep in departure_posts[:5]:
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[Post {idx+1}] Confidence: {dep['confidence']:.0%}")
        print(f"  Signals: {', '.join(dep['signals'])}")
        print(f"  Text: {preview}")

    # Escalation risk
    print("\n" + "-" * 80)
    print("ESCALATION RISK DETECTION")
    print("-" * 80)

    escalation_posts = []
    for idx, text in enumerate(texts):
        escalation = kn.detect_escalation_risk(text)
        if escalation['detected']:
            escalation_posts.append((idx, text, escalation))

    print(f"Posts with escalation risk: {len(escalation_posts)} / {len(texts)} ({100*len(escalation_posts)/len(texts):.1f}%)")

    for idx, text, esc in escalation_posts[:5]:
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[Post {idx+1}] Risk Level: {esc['risk_level'].upper()}")
        print(f"  Signals: {', '.join(esc['signals'])}")
        print(f"  Text: {preview}")

    # Intensity distribution
    print("\n" + "-" * 80)
    print("INTENSITY DISTRIBUTION")
    print("-" * 80)

    intensity_counts = {"neutral": 0, "mild": 0, "moderate": 0, "strong": 0, "extreme": 0}
    for text in texts:
        intensity = kn.get_intensity(text)
        intensity_counts[intensity['label']] += 1

    for label, count in intensity_counts.items():
        pct = 100 * count / len(texts)
        bar = "#" * int(pct / 2)
        print(f"  {label:<10} {count:>4} ({pct:5.1f}%) {bar}")

    # Save detailed results
    output_path = r"C:\Users\User\The Prof\Keyneg\gm_keyneg_analysis.csv"

    # Create detailed output
    detailed_results = []
    for idx, (text, result) in enumerate(zip(texts, results)):
        departure = kn.detect_departure_intent(text)
        escalation = kn.detect_escalation_risk(text)
        intensity = kn.get_intensity(text)

        detailed_results.append({
            'post_index': idx + 1,
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'negativity_score': result['negativity_score'],
            'severity': score_to_severity(result['negativity_score']),
            'top_sentiment': result['top_sentiment'],
            'top_keywords': '; '.join([f"{k[0]}({k[1]:.2f})" for k in result['keywords'][:5]]),
            'categories': '; '.join(result['categories']),
            'departure_intent': departure['detected'],
            'departure_signals': '; '.join(departure['signals']) if departure['detected'] else '',
            'escalation_risk': escalation['risk_level'] if escalation['detected'] else 'none',
            'escalation_signals': '; '.join(escalation['signals']) if escalation['detected'] else '',
            'intensity_level': intensity['label'],
        })

    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nDetailed results saved to: {output_path}")

    # Save keywords summary
    keywords_path = r"C:\Users\User\The Prof\Keyneg\gm_keywords_summary.csv"
    keywords_df = pd.DataFrame(summary['top_keywords'])
    keywords_df.to_csv(keywords_path, index=False)
    print(f"Keywords summary saved to: {keywords_path}")

    # Save sentiments summary
    sentiments_path = r"C:\Users\User\The Prof\Keyneg\gm_sentiments_summary.csv"
    sentiments_df = pd.DataFrame(summary['top_sentiments'])
    sentiments_df.to_csv(sentiments_path, index=False)
    print(f"Sentiments summary saved to: {sentiments_path}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
