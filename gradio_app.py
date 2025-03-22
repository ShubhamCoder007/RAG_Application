import gradio as gr
from doc_ingest import *
from chat_history import handle_query
import os
import re
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from model import get_model

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = get_model()
BASE_DIR = "chroma_db"
os.makedirs(BASE_DIR, exist_ok=True)
chat_history=[]

# Vector Store Management
def get_available_vector_stores():
    try:
        return [d for d in os.listdir(BASE_DIR) 
               if os.path.isdir(os.path.join(BASE_DIR, d))]
    except FileNotFoundError:
        return []


def run(query, vector_path, uploaded_file, chat_history):
    print(query, vector_path, uploaded_file, chat_history)
    print("*"*50)
    vector_db = None
    if uploaded_file is not None:
        vector_path=uploaded_file.split("\\")[-1]
        text=extract_text(uploaded_file)
        chunk=chunk_text(text)
        vector_db = create_vector_store(chunk, "chroma_db/"+vector_path)
    if vector_db is None:
        vector_db = load_vector_store("chroma_db/"+vector_path)
    answer, updated_history = handle_query(query, chat_history, llm, vector_db)
    return answer, updated_history


demo = gr.Interface(
    fn=run,
    inputs=[
        "text",
        gr.Dropdown(
            label="Select Vector Store",
            choices=get_available_vector_stores(),
            interactive=True),
        gr.File(label="Upload File"), 
        gr.State(value=[])  # Initialize chat history
    ],
    outputs=[
        "text",
        gr.State()  # To maintain chat history state
    ]
)

# # Run the app
if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)



def process_uploaded_file(file):
    # Process document and create vector store
    print("File : ",file)
    text = extract_text(file.name)
    chunks = chunk_text(text)
    
    # Create vector store using filename
    fn = file.split("\\")[-1]
    vector_path = f"chroma_db/{fn}"
    create_vector_store(chunks, persist_directory=vector_path)
    
    print(f"File processed and stored at: {vector_path}")  # Print path to console
    return vector_path, get_available_vector_stores()

def run(query, vector_path, chat_history, file=None):
    fn = file.split("\\")[-1]
    vector_path = f"chroma_db/{fn}"
    vector_db = load_vector_store(vector_path)
    answer, updated_history = handle_query(query, chat_history, llm, vector_db)
    return answer, updated_history, os.path.abspath(vector_path)  # Return absolute path


if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)