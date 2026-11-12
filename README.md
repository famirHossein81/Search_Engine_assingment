# Persian Boolean & Phrase Search Engine over Irandoc

**Course:** Search Engines and Web Mining — Homework 1  
**Instructor:** Dr. Zojaji  
**TA:** Moin Hossieni  

This repository contains the complete implementation of a Persian information retrieval pipeline and Boolean search engine built over the Irandoc scientific collection (100 thesis and research proposals).

---

## Architecture & Features

1. **Document Loading & Parsing (Task 1):**
   - Parses `IranDoc_Data.json` and extracts structured documents (`doc_1` to `doc_100`).
   - Exports clean datasets to `parsed_documents.json` and `parsed_documents.csv`.
   - Index unit: Combined title and abstract.

2. **Tokenization Comparison (Task 2):**
   - Implements both a **Space-based tokenizer** and a **Subword BPE tokenizer** trained on the Irandoc corpus.
   - Evaluates vocabulary size, average tokens per document, longest token, OOV rate, ZWNJ handling, and misspellings.

3. **Persian Normalization (Task 3):**
   - Case folding for English letters.
   - Character standardization (`ي` → `ی`, `ك` → `ک`, `ة` → `ه`).
   - Persian/Arabic digit unification (`۰-۹` / `٠-٩` → `0-9`).
   - Removal of Arabic diacritics and Harakat.
   - ZWNJ (`\u200c`) normalization for verbal prefixes (`می/نمی`) and suffixes (`ها/تر/ترین/ضمایر`).
   - Punctuation stripping.

4. **Stop Words Policy (Task 4):**
   - Curated list of 50+ frequent Persian stop words.
   - Preserved in positional index to support precise phrase queries.

5. **Stemming & Lemmatization (Task 5):**
   - Rule-based suffix stripper with protected word exceptions (`ایران`, `تهران`, `دانشگاه`, etc.).
   - Lemmatization via Hazm.
   - Comparison of 10 sample words with over/under-stemming error analysis.

6. **Positional Inverted Index (Task 6):**
   - Postings structure: `term -> {doc_id: {'tf': freq, 'positions': [p1, p2, ...]}}`.
   - Serialized to `inverted_index.json` (3,642 unique terms, 16,580 postings, ~817 KB).

7. **Boolean Retrieval (Task 7):**
   - Supports `AND`, `OR`, `NOT`, and nested parentheses `()` using the Shunting-Yard algorithm and stack evaluation.

8. **Phrase Queries (Task 8):**
   - Strict positional intersection (`pos2 = pos1 + 1`).
   - Completely eliminates false positives where both words appear in separate sentences.

9. **Demo & Remote Kaggle Execution (Task 9):**
   - Runs interactively in the terminal or on Kaggle (`se-hw1-irandoc-search-engine.ipynb`).

---

## Quickstart

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Demo & Interactive Search
```bash
python search_engine.py
```
Or search a specific query directly:
```bash
python search_engine.py "هوش AND مصنوعی"
python search_engine.py "\"یادگیری ماشین\""
python search_engine.py "(عصبی OR فازی) AND پیش‌بینی"
```

### 3. Kaggle Remote Execution
The notebook `kaggle_deploy/se-hw1-irandoc-search-engine.ipynb` can be executed directly on Kaggle:
```bash
kaggle kernels push -p kaggle_deploy
kaggle kernels status amirhosseinthefathi/se-hw1-irandoc-search-engine
kaggle kernels output amirhosseinthefathi/se-hw1-irandoc-search-engine -p kaggle_output
```
