from pathlib import Path
import json
import pickle
import math
import os
from nltk.stem import PorterStemmer
import string
from collections import defaultdict
from collections import Counter

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
        for word in input_list:
            if word in stop_words:
                input_list.remove(word)
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

    def __add_document(self, doc_id, text):
        tokenized_text = text.split()
        c = Counter(tokenized_text)
        for token in tokenized_text:
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

        with open(file_path_index, 'wb') as f:
            pickle.dump(self.index, f)

        with open(file_path_docmap, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(file_path_freq, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        file_path_index = Path("~/Krish/RAG/rag-search-engine/cache/index.pkl").expanduser()
        file_path_docmap = Path("~/Krish/RAG/rag-search-engine/cache/docmap.pkl").expanduser()
        file_path_freq = Path("~/Krish/RAG/rag-search-engine/cache/term_frequencies.pkl").expanduser()

        with open(file_path_index, "rb") as f:
            self.index = pickle.load(f)
            # print("loading index")

        with open(file_path_docmap, "rb") as f:
            self.docmap = pickle.load(f)
            # print("loading docmap")

        with open(file_path_freq, "rb") as f:
            self.term_frequencies = pickle.load(f)
            # print("loading term_frequencies")

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

        print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")





