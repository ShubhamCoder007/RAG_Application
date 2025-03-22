from langchain_groq import ChatGroq

def get_model():
    llm=ChatGroq(groq_api_key="",model_name="Gemma2-9b-It")
    return llm
