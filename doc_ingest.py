from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import numpy as np
np.float_ = np.float64

# 1. Document Extraction (unchanged)
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
    else:
        with open(file_path, 'r') as f:
            text = f.read()
    return text

# 2. Text Chunking (unchanged)
def chunk_text(text, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# 3. Updated Embedding & Storage with Persistence
def create_vector_store(texts, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()  # Explicit save to disk
    print(f"Vector store contains {vector_store._collection.count()} documents")
    return vector_store

# 4. New Function: Load Existing Vector Store
def load_vector_store(persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"Vector store contains {vector_db._collection.count()} documents")
    return vector_db



# Usage
# document_text = extract_text(r"C:\Users\shubh\Desktop\Workspace\RAG\data\bhagavad-gita-in-english-source-file.pdf")
# chunks = chunk_text(document_text)
# vector_db = create_vector_store(chunks)