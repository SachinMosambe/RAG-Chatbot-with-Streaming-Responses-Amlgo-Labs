import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import faiss
import pickle


# Ensure output directories exist
os.makedirs("../vectordb", exist_ok=True)
os.makedirs("../chunks", exist_ok=True)


# Converting text data into chunks

loader = TextLoader("../data/dataset.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

chunks = text_splitter.split_documents(docs)

# Converting chunks into embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([chunk.page_content for chunk in chunks])

# Creating a FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

# Saving the index and chunks
faiss.write_index(index, "../vectordb/index.faiss")
with open("../chunks/chunks.pkl", "wb") as fp:
    pickle.dump(chunks,fp)











