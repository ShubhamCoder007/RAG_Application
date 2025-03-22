from doc_ingest import *
from query_process import handle_query, load_vector_store

# Initialize system
file_path = r"C:\Users\shubh\Desktop\Workspace\RAG\data\Koushiki-Mukherjee-HR.pdf"
fn = file_path.split("\\")[-1]
vector_path = f"./chroma_db/{fn}"
document_text = extract_text(file_path)
chunks = chunk_text(document_text)
vector_db = create_vector_store(chunks, vector_path)

#load vector store
# vector_db = load_vector_store(vector_path)

# Query example
q=" "
while q!="":
    # vector_path = input("Enter the path of the doc you want to query:")
    # fn = vector_path.split("\\")[-1]
    q=input("Enter the query: ")
    if q=="":
        break
    response = handle_query(q,vector_db)
    print("Response:\n",response)