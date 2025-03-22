from typing import List, Tuple
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 1. Create Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

# 2. Custom Prompt Template with History
template = """Use the following context and chat history to answer the question.
If you don't know the answer, say you don't know. Don't make up answers.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""
QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["chat_history", "context", "question"]
)

# 3. Modified Answer Generator
def generate_answer(query: str, context: str, chat_history: List[Tuple[str, str]]):
    # Format chat history for LLM
    history_str = "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])
    
    # Create input for model
    input_text = QA_PROMPT.format(
        chat_history=history_str[-1000:],  # Keep last 1000 chars
        context=context,
        question=query
    )
    
    # Generate answer
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )
    return hf_pipe(input_text)[0]['generated_text']

# 4. Updated Query Handler with Memory
def handle_query(query: str, chat_history: List[Tuple[str, str]]):
    # Use last 2 exchanges for contextual retrieval
    context_query = query
    if chat_history:
        last_interactions = " ".join([q + " " + a for q, a in chat_history[-2:]])
        context_query = f"{last_interactions} {query}"
    
    # Retrieve with contextual query
    relevant_docs = retrieve_chunks(context_query, vector_db)
    
    # Generate answer
    if relevant_docs:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        answer = generate_answer(query, context, chat_history)
    else:
        answer = web_search_agent(query)
    
    # Update chat history (keep last 5 exchanges)
    return answer, chat_history[-4:] + [(query, answer)]

# 5. Usage Example
chat_history = []
question = "What is RAG?"

# First interaction
answer, chat_history = handle_query(question, chat_history)
print(f"User: {question}\nBot: {answer}\n")

# Follow-up question
question = "How does it handle context?"
answer, chat_history = handle_query(question, chat_history)
print(f"User: {question}\nBot: {answer}")






def rephrase_query(query, history):
    prompt = f"""Rephrase this question using chat history:
    History: {history}
    Question: {query}
    Rephrased:"""
    return hf_pipe(prompt, max_length=60)[0]['generated_text']
    
# Use in handle_query:
context_query = rephrase_query(query, chat_history[-3:])




import json

def save_chat_history(user_id, chat_history):
    with open(f"{user_id}_history.json", "w") as f:
        json.dump(chat_history, f)

def load_chat_history(user_id):
    try:
        with open(f"{user_id}_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    




# Test sequence
history = []
q1 = "What is photosynthesis?"
a1, history = handle_query(q1, history)

q2 = "What are its main stages?"
a2, history = handle_query(q2, history)

q3 = "Explain the first stage"
a3, history = handle_query(q3, history)  # Should reference previous answers