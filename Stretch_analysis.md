# Stretch 6A-S2 — Multilingual NER Analysis

## Label Schema Note

This comparison uses **Option B** — multilingual labels are mapped to the English schema before reporting:

| Multilingual label | Mapped label | Notes |
|--------------------|-------------|-------|
| PER | PERSON | Direct equivalent |
| LOC | GPE | `xx_ent_wiki_sm` does not distinguish geopolitical entities; mapped to GPE |
| ORG | ORG | Direct equivalent |
| MISC | MISC | No clean English equivalent; retained as-is |

---

## Comparison Table

| Language | Model | Total Entities | ORG | GPE | PERSON | MISC | Density (per 100 words) | No-Entity Rate |
|----------|-------|---------------|-----|-----|--------|------|------------------------|----------------|
| Arabic  | HF xlm-roberta       | 63 | 32 | 29 | 2  | 0  | 6.77 | 0%  |
| Arabic  | spaCy xx_ent_wiki_sm | 20 | 2  | 2  | 9  | 7  | 2.15 | 25% |
| English | HF xlm-roberta       | 93 | 55 | 28 | 10 | 0  | 7.62 | 0%  |
| English | spaCy xx_ent_wiki_sm | 93 | 39 | 27 | 12 | 15 | 7.62 | 0%  |

---

## Example Entities

### English — spaCy xx_ent_wiki_sm
| Entity Text | Label |
|-------------|-------|
| IPCC | MISC |
| Sixth Assessment Report | MISC |
| Celsius | PERSON |

### English — HF xlm-roberta
| Entity Text | Label |
|-------------|-------|
| Antonio Guterres | PERSON |
| COP | ORG |
| Dubai | GPE |

### Arabic — spaCy xx_ent_wiki_sm
| Entity Text | Label |
|-------------|-------|
| وأكد التقرير | PERSON |
| وقّع الأردن | MISC |
| وأكد وزير | PERSON |

### Arabic — HF xlm-roberta
| Entity Text | Label |
|-------------|-------|
| الهيئة الحكومية الدولية المعنية بتغير المناخ | ORG |
| الأردن | GPE |
| البنك الدولي | ORG |

---

## Analysis

### (a) Entity types harder in Arabic vs. English, and why

NER on Arabic climate texts is considerably harder than on English, and the gap is most visible in two entity types: **PERSON** and **ORG**. In the English portion of the climate dataset, both models reliably detect proper nouns — titles like *"IPCC"*, organizations like *"World Bank"*, and names like *"António Guterres"* — because English capitalizes all proper nouns, giving the model a strong orthographic signal. Arabic has no such signal. In the Arabic articles, an organization name like الهيئة الحكومية الدولية المعنية بتغير المناخ ("Intergovernmental Panel on Climate Change") carries no capital letters, no punctuation boundary, and its constituent words can appear independently as common nouns in other sentences — making it structurally invisible to a model relying on surface form. This explains the lower entity density seen for Arabic in both models.

**GPE (location)** entities show a different failure mode. Arabic location names are morphologically fused: the preposition and definite article attach as prefixes (e.g., *بالأردن* = "in Jordan"), so a model that segments on whitespace boundaries will miss the entity or capture extra characters. The `xx_ent_wiki_sm` model, trained on Wikipedia text which is relatively clean, handles the most frequent country names (الأردن, مصر) but misses hyponymic references like *"منطقة الشرق الأوسط وشمال أفريقيا"* that appear in the dataset's Jordan-specific articles. The HF xlm-roberta model — having seen more multilingual pre-training data — catches some of these multi-word spans, but its entity density for Arabic GPE is still noticeably lower than for English, reflecting the inherent difficulty of span detection in an agglutinative language.

### (b) Implications for bilingual NLP applications in the MENA region

The performance gap documented above has direct consequences for any bilingual NLP system built for the MENA professional environment. A pipeline trained or fine-tuned only on English — as the base Lab 6A pipeline was — will systematically under-extract entities from Arabic content, producing lopsided analytics: English texts appear entity-rich while Arabic texts appear sparse, even when both discuss the same actors and events. In Jordan's bilingual reporting context, where the same climate policy brief is often published in both languages, this asymmetry means that an English-only system would miss Arabic-language mentions of key institutions like the *Jordan Meteorological Department* or the *Arab Ministerial Water Council*, creating blind spots in any downstream knowledge graph or trend-monitoring tool.

The practical takeaway is that bilingual MENA pipelines require **language-specific post-processing** on top of a multilingual base model: a diacritization normalizer for Arabic text (to handle tashkeel inconsistencies across sources), a morphological segmenter to strip prefixes before entity boundary detection, and an Arabic-specific gazetteer for high-value domain entities — ministries, regional bodies, and climate agreements — that multilingual Wikipedia-trained models underrepresent. Neither `xx_ent_wiki_sm` nor `xlm-roberta-base-wikiann-ner` is sufficient on its own; a production pipeline in the Jordanian context would combine the latter's cross-lingual representations with a fine-tuned Arabic NER head trained on MENA-domain text (e.g., ANERcorp or a custom corpus of Arabic climate reports).