import pickle
import os


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}


    def __add_document(self, doc_id, text):
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term not in self.index:
            return []
        return sorted(self.index[term])

    def build(self, movies):
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)