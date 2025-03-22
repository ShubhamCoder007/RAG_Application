from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

def generate_answer(query: str, context: str, chat_history: list, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use this context:
         {context}. If you don't know, or the answer is not present in the context don't answer by giving the response I'm sorry, but i'm unable to answer."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    resp = chain.invoke({
        "input": query,
        "context": context,
        "chat_history": chat_history
    }).content
    return resp
