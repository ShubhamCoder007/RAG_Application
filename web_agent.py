from duckduckgo_search import DDGS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from generate_answer import generate_answer

def web_search_agent(query, llm, max_results=3):
    print("Triggered web agent")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    
    if not results:
        return "No relevant information found through web search"
    
    context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    return generate_answer(query, context, [], llm)