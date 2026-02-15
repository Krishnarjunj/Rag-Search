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
    # Removing punctuations
    Punctuations = string.punctuation
    table = str.maketrans("", "", Punctuations)
    input_string = input_string.translate(table)
    return input_string

def filter_stopwords_stemming(input_list):
    path_stop_words = Path("~/Krish/RAG/rag-search-engine/data/stopwords.txt").expanduser()
    with open(path_stop_words, "r") as f:
        data = f.read()
    stop_words = data.splitlines()

    stemmer = PorterStemmer()

    if isinstance(input_list, str):
        input_list = input_list.split()

        # removing stop words
        #for word in input_list:
        #   if word in stop_words:
        #       input_list.remove(word)
        input_list = [word for word in input_list if word not in stop_words]
        # stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])

        res_str = ""
        for word in input_list:
            res_str += word
            res_str += " "
        res_str = res_str.rstrip()

        return res_str

    else:
        # removing stop words
        for word in input_list:
            if word in stop_words:
                input_list.remove(word)
        # stemming
        for i in range(len(input_list)):
            input_list[i] = stemmer.stem(input_list[i])
        return input_list


class InvertedIndex:
    def __init__ (self):
        self.index = defaultdict(list)
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokenized_text = text.split()
        self.doc_lengths[doc_id] = len(tokenized_text)
        c = Counter(tokenized_text)
        #for token in tokenized_text:
        #   self.index[token.lower()].append(doc_id)
        for token in c.keys():
            self.index[token.lower()].append(doc_id)
        tokenized_str = ' '.join(tokenized_text)
        self.term_frequencies[int(doc_id)] = c

    def get_documents(self, term):
        res_list = []
        res_list = self.index[term]
        return sorted(res_list)

    def build(self):
        file_path_json = Path("~/Krish/RAG/rag-search-engine/data/movies.json").expanduser()
        with open(file_path_json, 'r') as f:
            data_json = json.load(f)
        for items in data_json["movies"]:
            term = f"{items['title']} {items['description']}"
            term = remove_punctuation(term)
            term = filter_stopwords_stemming(term)
            self.__add_document(items["id"], term)
            self.docmap[items["id"]] = items

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

    def get_tf(self, doc_id, term):
        term = remove_punctuation(term)
        term = term.split()
        term = filter_stopwords_stemming(term)

        if len(term) > 1:
            raise Exception("Length of term more than one")

        freq_in_doc = self.term_frequencies[doc_id].get(term[0], 0)

        if freq_in_doc == 0:
            return 0

        else:
            return freq_in_doc

    def get_bm25_idf(self, term: str) -> float:
        term = remove_punctuation(term)
        term = term.split()
        term = filter_stopwords_stemming(term)
        term = term[0]

        N = len(self.docmap)
        df = len(set(self.index[term]))

        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return bm25_idf

        #print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_b):
        tf = self.get_tf(doc_id, term)

        #term = remove_punctuation(term)
        #term = term.split()
        #term = filter_stopwords_stemming(term)
        #term = term[0]

        avg = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg)

        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_tf

    def __get_avg_doc_length(self) ->float:
        N = len(self.docmap)
        Total = 0
        for i,j in self.doc_lengths.items():
            Total += j
        avg = Total / N
        return avg

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        BM25_Score = bm25_tf * bm25_idf

        return BM25_Score

    def bm25_search(self, query, limit):
        term = remove_punctuation(query)
        term = term.split()
        term = filter_stopwords_stemming(term)
        scores = {}

        for i in self.docmap:
            total = 0
            for token in term:
                bm_score = self.bm25(i, token)
                total += bm_score
            scores[i] = total

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        for rank, (doc_id, score) in enumerate(sorted_scores[:limit], start = 1):
            title = self.docmap[doc_id]["title"]
            print(f"{rank}. ({doc_id}) {title} - Score: {score:.2f}")
