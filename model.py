from langchain_groq import ChatGroq

def get_model():
    llm=ChatGroq(groq_api_key="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",model_name="Gemma2-9b-It", temperature=0)
    return llm