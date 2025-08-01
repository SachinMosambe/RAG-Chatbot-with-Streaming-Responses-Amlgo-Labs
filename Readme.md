
# RAG Chatbot – TinyLlama + FAISS + Streamlit

This project is a lightweight Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions over your own documents. It uses TinyLlama for language generation, FAISS for semantic search, and Streamlit for a simple interactive interface. The system extracts text from PDFs, builds an embedding index, and streams contextual answers with sources.

---

## Project Structure

```
├── app.py                      # Streamlit chatbot app
├── src/                        # Core backend logic
│   ├── generator.py            # TinyLlama-based response generator
│   ├── retriever.py            # FAISS-based retriever
│   └── pipeline.py             # RAG pipeline logic
├── notebooks/                  # Data processing scripts
│   ├── raw_data_preprocessing.py  # Extracts and cleans text from PDFs
│   └── preprocessing.py           # Splits text and creates FAISS index
├── data/                       # Folder for raw PDFs and cleaned text
├── chunks/                     # Stored text chunks (.pkl)
├── vectordb/                   # FAISS index files
├── offload/                    # Model offload folder
├── requirements.txt            # Project dependencies
├── .env, .gitignore, venv/     # Environment config and virtual environment
└── README.md                   # This file
```

---

## How the Project Works

1. **PDF Extraction**
   Text is extracted and cleaned from PDF files.

2. **Chunking & Embeddings**
   The cleaned text is split into overlapping chunks and embedded using a sentence transformer.

3. **Indexing with FAISS**
   Embeddings are stored in a FAISS index for fast similarity search.

4. **RAG Pipeline**
   On a query, relevant chunks are retrieved and passed to the TinyLlama model for answer generation.

5. **Streaming UI**
   A Streamlit app provides a user interface that displays responses in a streaming fashion with retrieved sources.

---

## Setup Instructions

### 1. Clone the Repository

```
git clone <repo-url>
cd <project-folder>
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

Use a virtual environment (`venv/`) if desired.

---

## Running the Full Pipeline

### Step 1: Add Your PDF Files

Place all your PDF documents into the `data/` folder.

### Step 2: Extract and Clean the Text

Run the script to extract raw text from PDFs.

```
python notebooks/raw_data_preprocessing.py
```

This creates `data/dataset.txt`.

### Step 3: Generate Chunks and Build the FAISS Index

```
python notebooks/preprocessing.py
```

This creates the required `chunks/` and `vectordb/` directories.

---

## Running the Chatbot

Once preprocessing is done, launch the Streamlit app:

```
streamlit run app.py
```

You’ll see:

* A question input field
* Streaming responses from the LLM
* Retrieved source text chunks used to generate the answer

---

## Model and Embedding Choices

* **Language Model:** TinyLlama-1.1B-Chat
  A lightweight, open-source model suitable for local CPU/GPU inference.

* **Embedding Model:** all-MiniLM-L6-v2
  A fast and efficient sentence transformer that balances speed and semantic accuracy.

* **Vector Index:** FAISS
  Enables fast nearest-neighbor search on dense vectors.

---

## Example Questions

You can try asking:

* What is the refund policy mentioned in the document?
* Summarize the eligibility criteria.
* What is the deadline for submitting the form?

The chatbot will return context-aware answers and also display the relevant document snippets used to generate them.

---


## Demo Video


[Watch Demo](https://drive.google.com/file/d/1TI5kXYmwm1uDqzPwHLe7ZOXk8Vv6pcOg/view?usp=sharing)

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.

---

