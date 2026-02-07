from pathlib import Path
import json
import pickle
from collections import defaultdict

class InvertedIndex:
    def __init__ (self):
        self.index = defaultdict(list)
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokenized_text = text.split()
        for token in tokenized_text:
            self.index[token].append(doc_id)

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
            self.__add_document(items["id"], term)

    def save(self):
        file_path_index = Path("~/Krish/RAG/rag-search-engine/cache/index.pkl").expanduser()
        file_path_docmap = Path("~/Krish/RAG/rag-search-engine/cache/docmap.pkl").expanduser()

        with open(file_path_index, 'wb') as f:
            pickle.dump(self.index, f)

        with open(file_path_docmap, 'wb') as f:
            pickle.dump(self.docmap, f)






