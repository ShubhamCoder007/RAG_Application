from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from doc_ingest import load_vector_store
from web_agent import web_search_agent

from model import get_model
llm = get_model()

# 1. Retrieve Relevant Chunks
def retrieve_chunks(query, vector_db, k=3, threshold = 0.5):
    print(f"retrieve_chunk: Vector store contains {vector_db._collection.count()} documents")
    docs = vector_db.similarity_search_with_score(query, k=k)
    print("Scores: ", [score for doc, score in docs])
    threshold = 0.5
    return [doc for doc, score in docs if score >= threshold]

# 2. Generate Answer - simple version
def generate_answer(query, context, model_name="google/flan-t5-base"):
    # hf_pipe = pipeline("text2text-generation", model=model_name)
    # llm = HuggingFacePipeline(pipeline=hf_pipe)
    # print("context: ",context)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt).content
    return response



def handle_query(query, llm, vector_db=load_vector_store()):
    print(f"\nProcessing query: {query}")
    
    # Try local retrieval first
    relevant_docs = retrieve_chunks(query, vector_db=vector_db)
    print(f"Found {len(relevant_docs)} relevant local documents")
    
    if not relevant_docs:
        print("No local results, triggering web search")
        return web_search_agent(query, llm)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return generate_answer(query, context)


# print("Resp: ",handle_query("what did krishna say"))