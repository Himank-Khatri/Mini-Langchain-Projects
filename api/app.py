import os
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from fastapi import FastAPI
from langserve import add_routes
import uvicorn 
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

qwen = OllamaLLM(model='qwen2:0.5b')
llama = OllamaLLM(model='llama3.2:1b')

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

add_routes(
    app,
    prompt1 | llama,
    path="/essay"
)

add_routes(
    app,
    prompt2 | qwen,
    path="/poem"
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)
