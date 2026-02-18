from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text):
        if text.strip("") == "":
            raise ValueError(f"Empty string")

        else:
            input_list = []
            input_list.append(text)

            encoded_list = self.model.encode(input_list)

            return encoded_list[0]


def embed_text(text):
    Obj = SemanticSearch()
    embedding = Obj.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

    return


def verify_model():
    Obj = SemanticSearch()
    print(f"Model loaded: {Obj.model}")
    print(f"Max sequence length: {Obj.model.max_seq_length}")
    return



