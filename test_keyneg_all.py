"""
KeyNeg Multi-Company Analysis
=============================
Analyze all company posts tables and compare.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import sqlite3
import sys
from pathlib import Path
import pandas as pd

# Add keyneg to path
sys.path.insert(0, str(Path(__file__).parent))

from keyneg import KeyNeg
from keyneg.utils import aggregate_batch_results, score_to_severity

def analyze_company(kn, cursor, table_name):
    """Analyze a single company's posts."""
    # Get columns
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]

    # Find content column
    content_col = None
    for name in ['content', 'body', 'text', 'message', 'post', 'comment', 'post_content']:
        if name in col_names:
            content_col = name
            break

    if not content_col:
        return None

    # Get posts
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    total = cursor.fetchone()[0]

    limit = min(500, total)
    cursor.execute(f"SELECT {content_col} FROM {table_name} WHERE {content_col} IS NOT NULL AND {content_col} != '' LIMIT {limit};")
    posts = cursor.fetchall()

    if not posts:
        return None

    texts = [p[0] for p in posts]

    # Analyze
    results = kn.analyze_batch(texts, show_progress=False)
    summary = aggregate_batch_results(results)

    # Departure and escalation counts
    departure_count = sum(1 for t in texts if kn.detect_departure_intent(t)['detected'])
    escalation_count = sum(1 for t in texts if kn.detect_escalation_risk(t)['detected'])

    return {
        'table': table_name,
        'company': table_name.replace('_posts', '').replace('_', ' ').title(),
        'total_posts': total,
        'analyzed': len(texts),
        'avg_negativity': summary['avg_negativity_score'],
        'max_negativity': summary['max_negativity_score'],
        'std_negativity': summary['std_negativity_score'],
        'top_sentiment': summary['top_sentiments'][0]['sentiment'] if summary['top_sentiments'] else 'N/A',
        'top_keyword': summary['top_keywords'][0]['keyword'] if summary['top_keywords'] else 'N/A',
        'departure_pct': 100 * departure_count / len(texts),
        'escalation_pct': 100 * escalation_count / len(texts),
        'full_summary': summary
    }


def main():
    db_path = r"C:\Users\User\The Prof\thelayoff_scraper\thelayoff_data.db"

    print("=" * 80)
    print("KeyNeg Multi-Company Analysis")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all posts tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    posts_tables = [t for t in tables if '_posts' in t and 'wayback' not in t]

    print(f"\nFound {len(posts_tables)} company posts tables")
    print(f"Tables: {posts_tables}")

    # Initialize KeyNeg
    print("\nLoading KeyNeg model...")
    kn = KeyNeg(model="all-mpnet-base-v2")
    print(f"Model loaded: {kn}")

    # Analyze each company
    results = []
    for table in posts_tables:
        print(f"\nAnalyzing {table}...")
        result = analyze_company(kn, cursor, table)
        if result:
            results.append(result)
            print(f"  - Posts: {result['analyzed']}, Avg Negativity: {result['avg_negativity']:.3f}")

    # Sort by negativity
    results.sort(key=lambda x: x['avg_negativity'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPANY COMPARISON - SORTED BY NEGATIVITY")
    print("=" * 80)
    print(f"\n{'Company':<15} {'Posts':>6} {'Avg Neg':>8} {'Max Neg':>8} {'Depart%':>8} {'Escal%':>8} {'Top Sentiment':<25}")
    print("-" * 90)

    for r in results:
        print(f"{r['company']:<15} {r['analyzed']:>6} {r['avg_negativity']:>8.3f} {r['max_negativity']:>8.3f} {r['departure_pct']:>7.1f}% {r['escalation_pct']:>7.1f}% {r['top_sentiment']:<25}")

    # Top keywords across all companies
    print("\n" + "=" * 80)
    print("TOP SENTIMENTS BY COMPANY")
    print("=" * 80)

    for r in results:
        print(f"\n{r['company']}:")
        for s in r['full_summary']['top_sentiments'][:5]:
            print(f"  - {s['sentiment']:<30} (count: {s['count']}, avg: {s['avg_score']:.3f})")

    # Category comparison
    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTION BY COMPANY")
    print("=" * 80)

    all_categories = set()
    for r in results:
        for cat in r['full_summary']['category_distribution']:
            all_categories.add(cat['category'])

    # Build category matrix
    print(f"\n{'Category':<35}", end="")
    for r in results:
        print(f"{r['company'][:8]:>10}", end="")
    print()
    print("-" * (35 + 10 * len(results)))

    for cat in sorted(all_categories):
        print(f"{cat:<35}", end="")
        for r in results:
            count = 0
            for c in r['full_summary']['category_distribution']:
                if c['category'] == cat:
                    count = c['count']
                    break
            print(f"{count:>10}", end="")
        print()

    # Save to CSV
    df = pd.DataFrame([{
        'Company': r['company'],
        'Total Posts': r['total_posts'],
        'Analyzed': r['analyzed'],
        'Avg Negativity': r['avg_negativity'],
        'Max Negativity': r['max_negativity'],
        'Std Negativity': r['std_negativity'],
        'Top Sentiment': r['top_sentiment'],
        'Top Keyword': r['top_keyword'],
        'Departure %': r['departure_pct'],
        'Escalation %': r['escalation_pct']
    } for r in results])

    output_path = r"C:\Users\User\The Prof\Keyneg\keyneg_company_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    conn.close()
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
