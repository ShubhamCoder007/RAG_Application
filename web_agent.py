from duckduckgo_search import DDGS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from model import get_model
llm = get_model()


def generate_answer(query, context, model_name="google/flan-t5-base"):
    # hf_pipe = pipeline("text2text-generation", model=model_name)
    # llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    return llm(prompt).content


def web_search_agent(query, max_results=3):
    print("Triggered web agent")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    
    if not results:
        return "No relevant information found through web search"
    
    context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    return generate_answer(query, context)