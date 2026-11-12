#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian Boolean & Phrase Search Engine over Irandoc Dataset
Search Engines and Web Mining - Homework 1
Instructor: Dr. Zojaji | TA: Moin Hossieni
"""

import os
import sys
import json
import re
import time
import string
import csv
from collections import defaultdict, Counter

# Ensure UTF-8 output in Windows terminals
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Optional dependency: Hazm
try:
    import hazm
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False


# ==========================================
# 1. Normalization Pipeline
# ==========================================
class PersianNormalizer:
    def __init__(self):
        self.char_map = {
            'ي': 'ی', 'ى': 'ی', 'ك': 'ک', 'ة': 'ه',
            'ؤ': 'و', 'إ': 'ا', 'أ': 'ا', 'ء': '',
        }
        self.diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670]')
        punct = re.escape(string.punctuation + '،؛؟«»“”\n\r\t')
        self.punctuation_pattern = re.compile(f'[{punct}]')
        
        ar_digits = '٠١٢٣٤٥٦٧٨٩'
        fa_digits = '۰۱۲۳۴۵۶۷۸۹'
        en_digits = '0123456789'
        self.digit_trans = str.maketrans(ar_digits + fa_digits, en_digits * 2)

    def normalize(self, text: str) -> str:
        if not text:
            return ''
        # Case folding for any Latin fragments
        text = text.lower()
        # Unify Arabic chars to standard Persian
        for ar, fa in self.char_map.items():
            text = text.replace(ar, fa)
        # Digits standardization
        text = text.translate(self.digit_trans)
        # Remove diacritics / Harakat
        text = self.diacritics_pattern.sub('', text)
        # Standardize ZWNJ (half-space: \u200c)
        text = re.sub(r'\b(می|نمی)\s+', '\\1\u200c', text)
        text = re.sub(r'\s+(ها|های|تر|ترین|ام|ات|اش|ایم|اید|اند)\b', '\u200c\\1', text)
        text = re.sub(r'\u200c+', '\u200c', text)
        text = re.sub(r'\s*\u200c\s*', '\u200c', text)
        # Replace punctuation marks with whitespace
        text = self.punctuation_pattern.sub(' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ==========================================
# 2. Tokenizers (Space-based and Subword BPE)
# ==========================================
def space_tokenizer(text: str):
    return text.split()


class MiniBPE:
    def __init__(self, target_vocab_size=1200):
        self.target_vocab_size = target_vocab_size
        self.merges = {}
        self.vocab = set()

    def get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def train(self, texts):
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for w in words:
                chars = ' '.join(list(w)) + ' </w>'
                word_freqs[chars] += 1
                for c in list(w):
                    self.vocab.add(c)
        self.vocab.add('</w>')

        num_merges = self.target_vocab_size - len(self.vocab)
        for _ in range(max(0, num_merges)):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges[best] = ''.join(best)
            self.vocab.add(''.join(best))
            
            pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(best)) + r'(?!\S)')
            replacement = ''.join(best)
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_w = pattern.sub(replacement, word)
                new_word_freqs[new_w] += freq
            word_freqs = new_word_freqs

    def tokenize_word(self, word):
        symbols = list(word) + ['</w>']
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair_to_merge:
                    new_symbols.append(symbols[i] + symbols[i+1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def tokenize(self, text):
        words = text.split()
        subwords = []
        for w in words:
            subwords.extend(self.tokenize_word(w))
        return subwords


# ==========================================
# 3. Stop Words & Stemming
# ==========================================
PERSIAN_STOP_WORDS = set([
    'در', 'به', 'از', 'که', 'این', 'را', 'با', 'است', 'برای', 'آن', 'یک', 'شود', 'شده',
    'خود', 'ها', 'های', 'بر', 'تا', 'کرد', 'هم', 'نیز', 'اما', 'یا', 'وی',
    'بود', 'شد', 'می', 'کند', 'دارد', 'بین', 'پس', 'اگر', 'همه', 'صورت', 'یکی', 'دیگر',
    'روی', 'مورد', 'آنها', 'کرده', 'باشد', 'توسط', 'جهت', 'چون', 'چند', 'همچنین', 'باید',
    'هر', 'قبل', 'بعد', 'بوده', 'نمی', 'میان', 'درباره', 'زیرا', 'داشت'
])


class RuleBasedPersianStemmer:
    def __init__(self):
        self.suffixes = [
            '‌ترین', 'ترین', '‌تری', 'تری', '‌تر', 'تر',
            '‌گانه', 'گانه', '‌واره', '‌وار', 'واره', 'وار',
            '‌شناس', '‌شناسان', '‌شناسی',
            '‌مند', 'مند', '‌ناک', 'ناک',
            '‌های', 'های', '‌ها', 'ها',
            '‌ات', 'ات', '‌ان', 'ان', '‌ین', 'ین',
            '‌مان', '‌تان', '‌شان', '‌ام', '‌ات', '‌اش',
            '‌گی', '‌گری', 'گری',
            '‌ی', 'ی'
        ]
        self.exceptions = {
            'ایران', 'تهران', 'استان', 'داستان', 'روان', 'آسمان',
            'عنوان', 'قانون', 'بنیان', 'پژوهشگاه', 'پایگاه', 'دانشگاه', 'انسان', 'کیهان'
        }

    def stem(self, word: str) -> str:
        w = word.strip()
        if len(w) <= 3 or w in self.exceptions:
            return w
        for suffix in self.suffixes:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[:-len(suffix)]
                w = w.rstrip('\u200c')
                break
        return w


# ==========================================
# 4. Positional Inverted Index
# ==========================================
class PositionalInvertedIndex:
    def __init__(self, normalizer, stemmer, stop_words, keep_stop_words=True):
        self.normalizer = normalizer
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.keep_stop_words = keep_stop_words
        self.index = defaultdict(lambda: defaultdict(lambda: {'tf': 0, 'positions': []}))
        self.doc_metadata = {}
        self.doc_lengths = {}
        self.all_doc_ids = set()

    def build(self, documents):
        for doc in documents:
            doc_id = doc['doc_id']
            self.all_doc_ids.add(doc_id)
            self.doc_metadata[doc_id] = {
                'title': doc['title'],
                'authors': doc.get('authors', ''),
                'year': doc.get('year', '')
            }
            text = doc['full_text']
            norm_text = self.normalizer.normalize(text)
            raw_tokens = norm_text.split()
            self.doc_lengths[doc_id] = len(raw_tokens)
            
            for pos, raw_token in enumerate(raw_tokens):
                if not self.keep_stop_words and raw_token in self.stop_words:
                    continue
                term = self.stemmer.stem(raw_token)
                if not term:
                    continue
                entry = self.index[term][doc_id]
                entry['tf'] += 1
                entry['positions'].append(pos)

    def get_postings(self, term):
        norm_term = self.normalizer.normalize(term)
        stem_term = self.stemmer.stem(norm_term)
        return self.index.get(stem_term, {})

    def save(self, filepath):
        serializable = {
            'vocab_size': len(self.index),
            'total_postings': sum(len(postings) for postings in self.index.values()),
            'index': {k: dict(v) for k, v in self.index.items()},
            'doc_metadata': self.doc_metadata
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.index = defaultdict(lambda: defaultdict(lambda: {'tf': 0, 'positions': []}))
        for term, postings in data['index'].items():
            for doc_id, pdata in postings.items():
                self.index[term][doc_id] = pdata
        self.doc_metadata = data['doc_metadata']
        self.all_doc_ids = set(self.doc_metadata.keys())

    def stats(self):
        vocab_size = len(self.index)
        total_postings = sum(len(postings) for postings in self.index.values())
        return {
            'vocab_size': vocab_size,
            'total_postings': total_postings
        }


# ==========================================
# 5. Boolean & Phrase Retrieval Engine
# ==========================================
class BooleanSearchEngine:
    def __init__(self, index: PositionalInvertedIndex):
        self.index = index
        self.all_docs = set(index.all_doc_ids)

    def search_phrase(self, phrase: str):
        norm_phrase = self.index.normalizer.normalize(phrase)
        tokens = norm_phrase.split()
        if not tokens:
            return set()
        if len(tokens) == 1:
            return set(self.index.get_postings(tokens[0]).keys())
        
        postings = [self.index.get_postings(t) for t in tokens]
        for p in postings:
            if not p:
                return set()
                
        common_candidates = set(postings[0].keys())
        for p in postings[1:]:
            common_candidates.intersection_update(set(p.keys()))
            
        phrase_docs = set()
        for doc_id in common_candidates:
            first_positions = postings[0][doc_id]['positions']
            for p0 in first_positions:
                match = True
                for offset, p_other in enumerate(postings[1:], 1):
                    if (p0 + offset) not in p_other[doc_id]['positions']:
                        match = False
                        break
                if match:
                    phrase_docs.add(doc_id)
                    break
        return phrase_docs

    def tokenize_query(self, query: str):
        query = query.strip()
        # Clean stray backslashes from shell escaping (e.g. \"phrase\")
        query = re.sub(r'\\+"', '"', query)
        query = re.sub(r'"\\+', '"', query)
        # Normalize Persian quotation marks to standard double quotes
        query = re.sub(r'[«“]', '"', query)
        query = re.sub(r'[»”]', '"', query)
        
        raw_tokens = re.findall(r'\"[^\"]+\"|\(|\)|AND|OR|NOT|[^\s()]+', query)
        tokens = [t.strip() for t in raw_tokens if t.strip()]
        
        # Insert implicit AND between adjacent operands
        refined_tokens = []
        for i, t in enumerate(tokens):
            refined_tokens.append(t)
            if i < len(tokens) - 1:
                next_t = tokens[i+1]
                t_upper = t.upper()
                next_upper = next_t.upper()
                is_curr_operand = (t_upper not in ['AND', 'OR', 'NOT', '('])
                is_next_operand = (next_upper not in ['AND', 'OR', ')'])
                if is_curr_operand and is_next_operand:
                    refined_tokens.append('AND')
                    
        return refined_tokens

    def infix_to_postfix(self, tokens):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        operator_stack = []
        for t in tokens:
            upper_t = t.upper()
            if upper_t in ['AND', 'OR', 'NOT']:
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence[upper_t]):
                    output.append(operator_stack.pop())
                operator_stack.append(upper_t)
            elif t == '(':
                operator_stack.append(t)
            elif t == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                output.append(t)
        while operator_stack:
            output.append(operator_stack.pop())
        return output

    def evaluate_postfix(self, postfix_tokens):
        stack = []
        for token in postfix_tokens:
            upper_token = token.upper()
            if upper_token == 'NOT':
                if not stack:
                    continue
                s = stack.pop()
                stack.append(self.all_docs - s)
            elif upper_token == 'AND':
                if len(stack) < 2:
                    continue
                s2 = stack.pop()
                s1 = stack.pop()
                stack.append(s1.intersection(s2))
            elif upper_token == 'OR':
                if len(stack) < 2:
                    continue
                s2 = stack.pop()
                s1 = stack.pop()
                stack.append(s1.union(s2))
            else:
                if token.startswith('"') and token.endswith('"'):
                    phrase_str = token[1:-1]
                    res = self.search_phrase(phrase_str)
                else:
                    postings = self.index.get_postings(token)
                    res = set(postings.keys())
                stack.append(res)
        return stack.pop() if stack else set()

    def search(self, query: str):
        tokens = self.tokenize_query(query)
        postfix = self.infix_to_postfix(tokens)
        result_doc_ids = self.evaluate_postfix(postfix)
        sorted_ids = sorted(list(result_doc_ids), key=lambda x: int(x.split('_')[1]))
        results = []
        for did in sorted_ids:
            results.append({
                'doc_id': did,
                'title': self.index.doc_metadata[did]['title']
            })
        return results


# ==========================================
# 6. Interactive CLI & Demo Entrypoint
# ==========================================
def run_demo(search_engine):
    print("=" * 60)
    print("DEMO: Persian Boolean & Phrase Search Engine (Irandoc Dataset)")
    print("=" * 60)
    
    test_queries = [
        'هوش AND مصنوعی',
        'شبکه AND عصبی',
        'بازیابی AND NOT وب',
        '(عصبی OR فازی) AND پیش‌بینی',
        '"هوش مصنوعی"',
        '"شبکه عصبی"',
        '"یادگیری ماشین"'
    ]
    
    for q in test_queries:
        t0 = time.time()
        matches = search_engine.search(q)
        latency = (time.time() - t0) * 1000
        print(f"\n[Query] {q}")
        print(f"  -> Hits: {len(matches)} documents ({latency:.2f} ms)")
        for item in matches[:3]:
            print(f"     [{item['doc_id']}] {item['title'][:75]}...")


def main():
    normalizer = PersianNormalizer()
    stemmer = RuleBasedPersianStemmer()
    index = PositionalInvertedIndex(normalizer, stemmer, PERSIAN_STOP_WORDS, keep_stop_words=True)
    
    index_file = "inverted_index.json"
    data_file = "IranDoc_Data.json"
    
    if os.path.exists(index_file):
        print(f"Loading existing inverted index from '{index_file}'...")
        index.load(index_file)
    elif os.path.exists(data_file):
        print(f"Building index from '{data_file}'...")
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_docs = json.load(f)
        docs = []
        for idx, d in enumerate(raw_docs):
            t = (d.get('title') or '').strip()
            a = (d.get('abstract') or '').strip()
            docs.append({
                'doc_id': f"doc_{idx+1}",
                'title': t,
                'authors': d.get('authors', ''),
                'year': d.get('publication_year', ''),
                'full_text': f"{t} {a}".strip()
            })
        index.build(docs)
        index.save(index_file)
    else:
        print(f"Error: Neither '{index_file}' nor '{data_file}' was found.")
        return

    search_engine = BooleanSearchEngine(index)
    
    if len(sys.argv) > 1:
        raw_args = sys.argv[1:]
        if raw_args[0] in ['--phrase', '-p']:
            phrase_str = " ".join(raw_args[1:]).strip(' "\'«»“”\\')
            query = f'"{phrase_str}"'
        elif raw_args[0] in ['--query', '-q']:
            query = " ".join(raw_args[1:])
        else:
            query = " ".join(raw_args)
            
        results = search_engine.search(query)
        print(f"\nQuery: {query}")
        print(f"Total Matches: {len(results)}")
        for r in results:
            print(f"[{r['doc_id']}] {r['title']}")
        return

    # Run initial benchmark demo
    run_demo(search_engine)
    
    # Interactive loop
    print("\n" + "=" * 60)
    print("Interactive Mode. Type a query (or 'exit' to quit):")
    print("Supported: AND, OR, NOT, (), and \"exact phrases\"")
    print("=" * 60)
    while True:
        try:
            q = input("\nSearch > ").strip()
            if not q or q.lower() in ['exit', 'quit', 'q']:
                break
            t0 = time.time()
            res = search_engine.search(q)
            ms = (time.time() - t0) * 1000
            print(f"Results: {len(res)} hits ({ms:.2f} ms)")
            for i, r in enumerate(res[:10], 1):
                print(f"  {i}. [{r['doc_id']}] {r['title']}")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


if __name__ == '__main__':
    main()
