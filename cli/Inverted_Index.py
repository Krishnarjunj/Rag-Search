from pathlib import Path
import json
import pickle
import math
import os
from nltk.stem import PorterStemmer
import string
from collections import defaultdict
from collections import Counter

BM25_K1 = 1.5
BM25_b = 0.75

def remove_punctuation(input_string):
    table = str.maketrans("", "", string.punctuation)
    return input_string.translate(table)

def filter_stopwords_stemming(input_list):
    path_stop_words = Path("~/Krish/RAG/rag-search-engine/data/stopwords.txt").expanduser()
    with open(path_stop_words, "r") as f:
        stop_words = set(f.read().splitlines())

    stemmer = PorterStemmer()

    # Ensure we are working with a list of tokens
    if isinstance(input_list, str):
        tokens = input_list.split()
    else:
        tokens = input_list

    # Fix: Use list comprehension to avoid skipping words during iteration
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    if isinstance(input_list, str):
        return " ".join(cleaned_tokens)
    return cleaned_tokens

class InvertedIndex:
    def __init__ (self):
        self.index = defaultdict(list)
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, processed_text_str):
        # We split the already processed string to ensure lengths match the Counter
        tokens = processed_text_str.split()
        self.doc_lengths[doc_id] = len(tokens)
        
        counts = Counter(tokens)
        for token in counts.keys():
            # Using doc_id as provided (ensure types are consistent)
            self.index[token.lower()].append(doc_id)
        
        self.term_frequencies[doc_id] = counts

    def build(self):
        file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
        with open(file_path_json, 'r') as f:
            data_json = json.load(f)
            
        for items in data_json["movies"]:
            doc_id = items["id"]
            combined_text = f"{items['title']} {items['description']}"
            
            # Standardize processing pipeline
            clean_text = remove_punctuation(combined_text).lower()
            processed_str = filter_stopwords_stemming(clean_text)
            
            self.__add_document(doc_id, processed_str)
            self.docmap[doc_id] = items

    def get_tf(self, doc_id, term):
        # Process the single query term the same way as indexing
        clean_term = remove_punctuation(term).lower()
        processed_tokens = filter_stopwords_stemming(clean_term)
        
        # filter_stopwords_stemming returns a string if input was string
        target = processed_tokens.split()[0] if processed_tokens else ""
        
        # Ensure doc_id lookup matches the type stored (int vs str)
        doc_freqs = self.term_frequencies.get(doc_id, {})
        return doc_freqs.get(target, 0)

    def get_bm25_idf(self, term: str) -> float:
        clean_term = remove_punctuation(term).lower()
        processed = filter_stopwords_stemming(clean_term).split()
        if not processed: return 0.0
        target = processed[0]

        N = len(self.docmap)
        # DF is the number of documents containing the term
        df = len(set(self.index.get(target, [])))

        # Standard BM25 IDF formula
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_b):
        tf = self.get_tf(doc_id, term)
        if tf == 0: return 0.0

        avg_dl = sum(self.doc_lengths.values()) / len(self.docmap)
        doc_len = self.doc_lengths.get(doc_id, 0)
        
        # Length normalization component
        L_d = 1 - b + b * (doc_len / avg_dl)
        
        return (tf * (k1 + 1)) / (tf + k1 * L_d)

    def bm25_search(self, query, limit):
        # Process query into tokens
        clean_query = remove_punctuation(query).lower()
        query_tokens = filter_stopwords_stemming(clean_query).split()
        
        scores = Counter()
        for doc_id in self.docmap:
            total_score = 0
            for token in query_tokens:
                # Calculate score per token
                idf = self.get_bm25_idf(token)
                tf_score = self.get_bm25_tf(doc_id, token)
                total_score += (idf * tf_score)
            if total_score > 0:
                scores[doc_id] = total_score

        sorted_scores = scores.most_common(limit)

        for rank, (doc_id, score) in enumerate(sorted_scores, start=1):
            title = self.docmap[doc_id]["title"]
            print(f"{rank}. ({doc_id}) {title} - Score: {score:.2f}")

    def save(self):
        file_path_index = Path("~/Krish/RAG/rag-search-engine/cache/index.pkl").expanduser()
        file_path_docmap = Path("~/Krish/RAG/rag-search-engine/cache/docmap.pkl").expanduser()
        file_path_freq = Path("~/Krish/RAG/rag-search-engine/cache/term_frequencies.pkl").expanduser()
        file_path_doclen = Path("~/Krish/RAG/rag-search-engine/cache/doc_lenghts.pkl").expanduser()

        with open(file_path_index, 'wb') as f:
            pickle.dump(self.index, f)

        with open(file_path_docmap, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(file_path_freq, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(file_path_doclen, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        file_path_index = Path("~/Krish/RAG/rag-search-engine/cache/index.pkl").expanduser()
        file_path_docmap = Path("~/Krish/RAG/rag-search-engine/cache/docmap.pkl").expanduser()
        file_path_freq = Path("~/Krish/RAG/rag-search-engine/cache/term_frequencies.pkl").expanduser()
        file_path_doclen = Path("~/Krish/RAG/rag-search-engine/cache/doc_lenghts.pkl").expanduser()

        with open(file_path_index, "rb") as f:
            self.index = pickle.load(f)
            # print("loading index")

        with open(file_path_docmap, "rb") as f:
            self.docmap = pickle.load(f)
            # print("loading docmap")

        with open(file_path_freq, "rb") as f:
            self.term_frequencies = pickle.load(f)
            # print("loading term_frequencies")

        with open(file_path_doclen, "rb") as f:
            self.doc_lengths = pickle.load(f)












