# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
# from model import get_model
from query_process import retrieve_chunks, web_search_agent
from doc_ingest import load_vector_store
from generate_answer import generate_answer


def handle_query(query: str, chat_history: list, llm, vector_db=load_vector_store()):
    # Use last 2 messages for contextual retrieval
    context_query = " ".join([
        msg.content for msg in chat_history[-2:]
    ]) + " " + query
    
    # Retrieve documents
    relevant_docs = retrieve_chunks(context_query, vector_db)
    
    # Generate answer
    if relevant_docs:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        answer = generate_answer(query, context, chat_history, llm)
    else:
        answer = web_search_agent(query, llm)
    
    # Update chat history
    updated_history = chat_history + [
        HumanMessage(content=query),
        AIMessage(content=answer)
    ][-3:]

    print(f"Updated history: {len(updated_history)}")
    
    return answer, updated_history
