"""
Stretch 6A-S2 — Multilingual NER Comparison
Compares spaCy xx_ent_wiki_sm and HuggingFace xlm-roberta-base-wikiann-ner
across English and Arabic climate articles.

Run:
    python -m spacy download xx_ent_wiki_sm
    pip install transformers torch
    python stretch_multilingual_ner.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import spacy
from transformers import pipeline


# ---------------------------------------------------------------------------
# Label mapping: multilingual labels → English schema (Option B)
# Documented explicitly as required by the assignment.
#
#   Multilingual label  →  Mapped label   Notes
#   PER                 →  PERSON         Direct equivalent
#   LOC                 →  GPE            xx_ent_wiki_sm doesn't distinguish
#                                         geopolitical entities; mapped to GPE
#   ORG                 →  ORG            Direct equivalent
#   MISC                →  MISC           No clean English equivalent; kept
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "PER": "PERSON", "LOC": "GPE", "ORG": "ORG", "MISC": "MISC",
    # HF xlm-roberta uses BIO prefix — aggregation_strategy='simple' removes
    # them, but kept here as a safety net
    "B-PER": "PERSON", "I-PER": "PERSON",
    "B-LOC": "GPE",    "I-LOC": "GPE",
    "B-ORG": "ORG",    "I-ORG": "ORG",
    "B-MISC": "MISC",  "I-MISC": "MISC",
    "O": "MISC",
}

def map_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_texts(filepath="data/climate_articles.csv", n=20):
    """Load at least n English and n Arabic texts."""
    df = pd.read_csv(filepath)
    en = df[df["language"] == "en"].head(n).reset_index(drop=True)
    ar = df[df["language"] == "ar"].head(n).reset_index(drop=True)
    print(f"[Data] Loaded {len(en)} English + {len(ar)} Arabic texts")
    return en, ar


def word_count(text: str) -> int:
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Model 1 — spaCy xx_ent_wiki_sm
# ---------------------------------------------------------------------------
def run_spacy_multilingual(texts_df: pd.DataFrame) -> dict:
    nlp = spacy.load("xx_ent_wiki_sm")
    all_ents = []
    zero_count = 0

    for _, row in texts_df.iterrows():
        doc = nlp(str(row["text"]))
        found = [
            (row["id"], ent.text.strip(), map_label(ent.label_))
            for ent in doc.ents if ent.text.strip()
        ]
        if not found:
            zero_count += 1
        all_ents.extend(found)

    return _build_result(all_ents, zero_count, len(texts_df))


# ---------------------------------------------------------------------------
# Model 2 — HuggingFace xlm-roberta-base-wikiann-ner
# ---------------------------------------------------------------------------
def run_hf_multilingual(texts_df: pd.DataFrame) -> dict:
    ner = pipeline(
        "ner",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple",  # merges B-/I- tokens automatically
        device=-1,                       # CPU
    )
    all_ents = []
    zero_count = 0

    for _, row in texts_df.iterrows():
        text = str(row["text"])[:1000]   # stay within 512-token limit
        try:
            results = ner(text)
        except Exception:
            results = []

        found = [
            (row["id"], r["word"].strip(), map_label(r.get("entity_group", r.get("entity", ""))))
            for r in results if r["word"].strip()
        ]
        if not found:
            zero_count += 1
        all_ents.extend(found)

    return _build_result(all_ents, zero_count, len(texts_df))


def _build_result(all_ents, zero_count, total_texts):
    label_counts = {}
    for _, _, label in all_ents:
        label_counts[label] = label_counts.get(label, 0) + 1

    seen, examples = set(), []
    for _, ent_text, label in all_ents:
        if ent_text not in seen:
            examples.append((ent_text, label))
            seen.add(ent_text)
        if len(examples) == 3:
            break

    return {
        "entities": all_ents,
        "label_counts": label_counts,
        "examples": examples,
        "no_entity_rate": zero_count / total_texts if total_texts else 0,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(results: dict, texts: dict) -> pd.DataFrame:
    rows = []
    for (lang, model), res in results.items():
        total_words = sum(word_count(t) for t in texts[lang]["text"])
        total_ents  = len(res["entities"])
        density     = round((total_ents / total_words) * 100, 2) if total_words else 0
        lc = res["label_counts"]
        rows.append({
            "Language":               "English" if lang == "en" else "Arabic",
            "Model":                  model,
            "Total Entities":         total_ents,
            "ORG":                    lc.get("ORG", 0),
            "GPE":                    lc.get("GPE", 0),
            "PERSON":                 lc.get("PERSON", 0),
            "MISC":                   lc.get("MISC", 0),
            "Density (per 100 words)": density,
            "No-Entity Rate":         f"{res['no_entity_rate']:.0%}",
        })

    return (pd.DataFrame(rows)
              .sort_values(["Language", "Model"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Stretch 6A-S2 — Multilingual NER Comparison")
    print("Label schema: Option B — mapped to PERSON / GPE / ORG / MISC")
    print("=" * 70)

    en_texts, ar_texts = load_texts("data/climate_articles.csv", n=20)
    texts = {"en": en_texts, "ar": ar_texts}
    results = {}

    print("\n[spaCy xx_ent_wiki_sm] English …")
    results[("en", "spaCy xx_ent_wiki_sm")] = run_spacy_multilingual(en_texts)

    print("[spaCy xx_ent_wiki_sm] Arabic …")
    results[("ar", "spaCy xx_ent_wiki_sm")] = run_spacy_multilingual(ar_texts)

    print("\n[HF xlm-roberta] English …")
    results[("en", "HF xlm-roberta")] = run_hf_multilingual(en_texts)

    print("[HF xlm-roberta] Arabic …")
    results[("ar", "HF xlm-roberta")] = run_hf_multilingual(ar_texts)

    # --- Comparison table ---
    table = build_comparison_table(results, texts)
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string(index=False))
    table.to_csv("stretch_comparison_table.csv", index=False, encoding="utf-8-sig")
    print("\n[Output] stretch_comparison_table.csv saved")

    # --- Example entities ---
    print("\n" + "=" * 70)
    print("EXAMPLE ENTITIES (3 per Language × Model)")
    print("=" * 70)
    for (lang, model), res in results.items():
        print(f"\n  [{'English' if lang == 'en' else 'Arabic'}] {model}")
        for ent_text, label in (res["examples"] or [("(none)", "-")]):
            print(f"    • {ent_text:<40} [{label}]")

    print("\n[Done]")


if __name__ == "__main__":
    main()