from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")



def verify_model():
    Obj = SemanticSearch()
    print(f"Model loaded: {Obj.model}")
    print(f"Max sequence length: {Obj.model.max_seq_length}")



