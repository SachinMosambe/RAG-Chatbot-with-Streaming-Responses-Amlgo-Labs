from .retriever import Retriever
from .generator import Generator

class Pipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(self, query):
        chunks = self.retriever.retrieve(query)
        context = "\n".join(chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in chunks)
        answer = self.generator.generate(context, query)
        return answer, chunks

    def stream_response(self, query):
        chunks = self.retriever.retrieve(query)
        context = "\n".join(chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in chunks)

        # Simulate streaming (word-by-word or sentence-by-sentence)
        answer = self.generator.generate(context, query)
        for sentence in answer.split(". "):
            yield sentence + ". ", chunks

    def get_num_chunks(self):
        return len(self.retriever.chunks)
