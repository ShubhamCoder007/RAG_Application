from doc_ingest import *
from chat_history import handle_query
from model import get_model

# Initialize system
file_path = r"C:\Users\shubh\Desktop\Workspace\RAG\data\Koushiki-Mukherjee-HR.pdf"
fn = file_path.split("\\")[-1]
vector_path = f"./chroma_db/{fn}"
document_text = extract_text(file_path)
chunks = chunk_text(document_text)
vector_db = create_vector_store(chunks, vector_path)

#load vector store
# vector_db = load_vector_store(vector_path)

llm = get_model()

chat_history = []
q=" "
while q!="":
    q=input("Enter the query: ")
    if q=="":
        break
    answer, chat_history = handle_query(q, chat_history, llm, vector_db)
    print("Response:\n",answer)