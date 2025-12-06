"""
KeyNeg Test Script
==================
Test KeyNeg on thelayoff_data.db posts tables.

Author: Kaossara Osseni
Email: admin@grandnasser.com
"""

import sqlite3
import sys
from pathlib import Path

# Add keyneg to path
sys.path.insert(0, str(Path(__file__).parent))

from keyneg import KeyNeg
from keyneg.utils import aggregate_batch_results, score_to_severity, export_batch_to_csv

def main():
    # Connect to database
    db_path = r"C:\Users\User\The Prof\thelayoff_scraper\thelayoff_data.db"

    print("=" * 60)
    print("KeyNeg Test - Analyzing TheLayoff Posts")
    print("=" * 60)

    # Connect and explore
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [t[0] for t in tables]

    # Find posts tables
    posts_tables = [t for t in table_names if '_posts' in t and 'wayback' not in t]
    print(f"\nAvailable posts tables: {posts_tables}")

    # Use verizon_posts as example (or combine multiple)
    target_table = 'verizon_posts'
    if target_table not in table_names:
        target_table = posts_tables[0] if posts_tables else None

    if not target_table:
        print("No posts tables found!")
        conn.close()
        return

    print(f"\nAnalyzing table: {target_table}")

    # Check table structure
    cursor.execute(f"PRAGMA table_info({target_table});")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]
    print(f"Columns: {col_names}")

    # Count posts
    cursor.execute(f"SELECT COUNT(*) FROM {target_table};")
    total_posts = cursor.fetchone()[0]
    print(f"Total posts: {total_posts}")

    # Find content column
    content_col = None
    for name in ['content', 'body', 'text', 'message', 'post', 'comment', 'post_content']:
        if name in col_names:
            content_col = name
            break

    if not content_col:
        # Check column types
        for col in columns:
            if col[2].upper() in ['TEXT', 'VARCHAR', 'STRING']:
                content_col = col[1]
                break

    print(f"Using content column: {content_col}")

    # Get id column
    id_col = 'id' if 'id' in col_names else 'post_id' if 'post_id' in col_names else 'rowid'

    # Get posts for analysis
    limit = min(200, total_posts)
    query = f"SELECT {id_col}, {content_col} FROM {target_table} WHERE {content_col} IS NOT NULL AND {content_col} != '' LIMIT {limit};"
    cursor.execute(query)
    posts = cursor.fetchall()

    print(f"\nLoaded {len(posts)} posts for analysis")

    if not posts:
        print("No posts found to analyze!")
        conn.close()
        return

    # Show sample posts
    print("\n" + "-" * 60)
    print("Sample Posts:")
    print("-" * 60)
    for i, (post_id, content) in enumerate(posts[:3]):
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n[Post {post_id}]: {preview}")

    # Initialize KeyNeg
    print("\n" + "=" * 60)
    print("Initializing KeyNeg (loading MPNet model)...")
    print("=" * 60)

    kn = KeyNeg(model="all-mpnet-base-v2")
    print(f"KeyNeg loaded: {kn}")

    # Extract texts
    texts = [content for _, content in posts]
    post_ids = [post_id for post_id, _ in posts]

    # Batch analysis
    print(f"\nAnalyzing {len(texts)} posts...")
    results = kn.analyze_batch(texts, show_progress=True)

    # Aggregate results
    summary = aggregate_batch_results(results)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nDocuments analyzed: {summary['total_documents']}")
    print(f"Average negativity score: {summary['avg_negativity_score']:.3f}")
    print(f"Max negativity score: {summary['max_negativity_score']:.3f}")
    print(f"Min negativity score: {summary['min_negativity_score']:.3f}")
    print(f"Std deviation: {summary['std_negativity_score']:.3f}")

    # Top sentiments
    print("\n" + "-" * 60)
    print("TOP 10 SENTIMENTS (Most Frequent)")
    print("-" * 60)
    for item in summary['top_sentiments'][:10]:
        print(f"  {item['sentiment']:<35} Count: {item['count']:>4}  Avg Score: {item['avg_score']:.3f}")

    # Top keywords
    print("\n" + "-" * 60)
    print("TOP 15 KEYWORDS (Most Frequent)")
    print("-" * 60)
    for item in summary['top_keywords'][:15]:
        print(f"  {item['keyword']:<35} Count: {item['count']:>4}  Avg Score: {item['avg_score']:.3f}")

    # Category distribution
    print("\n" + "-" * 60)
    print("CATEGORY DISTRIBUTION")
    print("-" * 60)
    for item in summary['category_distribution']:
        print(f"  {item['category']:<40} Count: {item['count']:>4}")

    # Most negative posts
    print("\n" + "-" * 60)
    print("TOP 5 MOST NEGATIVE POSTS")
    print("-" * 60)

    # Sort by negativity score
    scored_posts = list(zip(post_ids, texts, results))
    scored_posts.sort(key=lambda x: x[2]['negativity_score'], reverse=True)

    for post_id, text, result in scored_posts[:5]:
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"\n[Post {post_id}] Negativity: {result['negativity_score']:.3f} ({score_to_severity(result['negativity_score']).upper()})")
        print(f"  Top Sentiment: {result['top_sentiment']}")
        print(f"  Categories: {', '.join(result['categories'][:3])}")
        print(f"  Text: {preview}")

    # Departure intent detection
    print("\n" + "-" * 60)
    print("DEPARTURE INTENT SIGNALS")
    print("-" * 60)

    departure_posts = []
    for post_id, text, result in zip(post_ids, texts, results):
        departure = kn.detect_departure_intent(text)
        if departure['detected']:
            departure_posts.append((post_id, text, departure))

    print(f"Posts with departure intent: {len(departure_posts)} / {len(texts)} ({100*len(departure_posts)/len(texts):.1f}%)")

    for post_id, text, departure in departure_posts[:3]:
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[Post {post_id}] Confidence: {departure['confidence']:.0%}")
        print(f"  Signals: {', '.join(departure['signals'])}")
        print(f"  Text: {preview}")

    # Escalation risk
    print("\n" + "-" * 60)
    print("ESCALATION RISK SIGNALS")
    print("-" * 60)

    escalation_posts = []
    for post_id, text, result in zip(post_ids, texts, results):
        escalation = kn.detect_escalation_risk(text)
        if escalation['detected']:
            escalation_posts.append((post_id, text, escalation))

    print(f"Posts with escalation risk: {len(escalation_posts)} / {len(texts)} ({100*len(escalation_posts)/len(texts):.1f}%)")

    for post_id, text, escalation in escalation_posts[:3]:
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[Post {post_id}] Risk Level: {escalation['risk_level'].upper()}")
        print(f"  Signals: {', '.join(escalation['signals'])}")
        print(f"  Text: {preview}")

    # Intensity analysis
    print("\n" + "-" * 60)
    print("INTENSITY DISTRIBUTION")
    print("-" * 60)

    intensity_counts = {"neutral": 0, "mild": 0, "moderate": 0, "strong": 0, "extreme": 0}
    for text in texts:
        intensity = kn.get_intensity(text)
        intensity_counts[intensity['label']] += 1

    for label, count in intensity_counts.items():
        pct = 100 * count / len(texts)
        bar = "#" * int(pct / 2)
        print(f"  {label:<10} {count:>4} ({pct:5.1f}%) {bar}")

    # Save results to CSV
    output_path = r"C:\Users\User\The Prof\Keyneg\keyneg_analysis_results.csv"
    export_batch_to_csv(results, output_path, include_keywords=True, max_keywords=5)
    print(f"\n\nResults saved to: {output_path}")

    conn.close()
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
