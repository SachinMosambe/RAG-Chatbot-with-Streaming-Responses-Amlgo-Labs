from sentence_transformers import SentenceTransformer
import faiss
import pickle

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("./vectordb/index.faiss")

        with open("./chunks/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, top_k=3):
        embedding = self.model.encode([query])
        distances, indices = self.index.search(embedding, top_k)
        return [self.chunks[i] for i in indices[0]]
