"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    return pd.read_csv(filepath)


def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    result = df.copy()

    def process_row(row):
        normalized = unicodedata.normalize("NFC", str(row["text"]))
        if row["language"] == "en":
            return normalized
        else:
            # Arabic (or any non-English): pass NFC text through without
            # attempting English NLP. Empty string is also acceptable.
            return normalized

    result["processed_text"] = result.apply(process_row, axis=1)
    return result


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_rows = df[df["language"] == "en"]

    records = []
    for _, row in english_rows.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            records.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(records, columns=["text_id", "entity_text", "entity_label",
                                          "start_char", "end_char"])


def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count
                           (or filter to co_count >= 2) so the result
                           stays readable on the full corpus.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    # --- Top 20 entities by frequency ---
    freq = (
        entity_df.groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # --- Label counts ---
    label_counts = (
        entity_df.groupby("entity_label")
        .size()
        .to_dict()
    )

    # --- Co-occurrence: pairs of entities in the same text ---
    co_records = []
    for text_id, group in entity_df.groupby("text_id"):
        unique_entities = group["entity_text"].unique().tolist()
        for a, b in combinations(sorted(unique_entities), 2):
            co_records.append((a, b))

    if co_records:
        co_df = (
            pd.DataFrame(co_records, columns=["entity_a", "entity_b"])
            .groupby(["entity_a", "entity_b"])
            .size()
            .reset_index(name="co_count")
            .sort_values("co_count", ascending=False)
        )
        # Cap at top 50 pairs (or co_count >= 2)
        co_df = co_df[co_df["co_count"] >= 2].head(50).reset_index(drop=True)
    else:
        co_df = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])

    # --- Per-category breakdown ---
    merged = entity_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )
    per_category = (
        merged.groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    stats = {
        "top_entities": freq,
        "label_counts": label_counts,
        "co_occurrence": co_df,
        "per_category": per_category,
    }

    # Console summary
    print(f"\n[Stats Summary]")
    print(f"  Total entity mentions : {len(entity_df)}")
    print(f"  Unique entity types   : {len(label_counts)}")
    print(f"  Top label             : {max(label_counts, key=label_counts.get)} "
          f"({max(label_counts.values())} mentions)")
    print(f"  Co-occurrence pairs   : {len(co_df)}")

    return stats


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    top = stats["top_entities"].copy()

    # Assign a color to each entity label
    unique_labels = top["entity_label"].unique()
    palette = plt.cm.get_cmap("tab20", len(unique_labels))
    color_map = {label: palette(i) for i, label in enumerate(unique_labels)}
    colors = top["entity_label"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(
        top["entity_text"][::-1],  # reverse so highest is on top
        top["count"][::-1],
        color=colors[::-1],
    )

    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Entity", fontsize=12)
    ax.set_title("Top 20 Named Entities in Climate Articles Corpus", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=color_map[lbl], label=lbl) for lbl in unique_labels]
    ax.legend(handles=legend_handles, title="Entity Type", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ENTITY ANALYSIS REPORT — Climate Articles Corpus")
    lines.append("=" * 60)

    # --- Entity counts per type ---
    lines.append("\n## Entity Counts by Type")
    label_counts = stats["label_counts"]
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<12} {count:>5} mentions")

    # --- Top 5 most frequent entities ---
    lines.append("\n## Top 5 Most Frequent Entities")
    top5 = stats["top_entities"].head(5)
    for _, row in top5.iterrows():
        lines.append(f"  {row['entity_text']:<30} [{row['entity_label']}]  {row['count']} mentions")

    # --- Top 3 co-occurring pairs ---
    lines.append("\n## Top 3 Co-occurring Entity Pairs")
    if co_occurrence is not None and len(co_occurrence) > 0:
        top3_co = co_occurrence.head(3)
        for _, row in top3_co.iterrows():
            lines.append(f"  '{row['entity_a']}' + '{row['entity_b']}'  →  {int(row['co_count'])} shared texts")
    else:
        lines.append("  No co-occurrence data available.")

    # --- Summary paragraph ---
    total_mentions = sum(label_counts.values())
    dominant_label = max(label_counts, key=label_counts.get)
    dominant_count = label_counts[dominant_label]
    top_entity = stats["top_entities"].iloc[0]["entity_text"] if len(stats["top_entities"]) > 0 else "N/A"

    lines.append("\n## Summary")
    lines.append(
        f"  The corpus contains {total_mentions} total entity mentions across "
        f"{len(label_counts)} entity types. The most common entity type is "
        f"{dominant_label} with {dominant_count} mentions, reflecting the prominent "
        f"role of organizations and institutions in climate discourse. The single most "
        f"frequently mentioned entity is '{top_entity}'. Co-occurrence patterns reveal "
        f"which actors and locations are discussed together, providing a structural map "
        f"of the relationships driving the climate narrative in this dataset."
    )

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)