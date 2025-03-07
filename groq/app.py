import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import time
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

@st.cache_data
def setup():
    st.session_state.embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    st.session_state.loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

if "vector" not in st.session_state:
    setup()


st.title("Chat with Langchain Docs")
llm = ChatGroq(groq_api_key=groq_api_key,
               model='llama-3.3-70b-versatile')

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
"""
)
    
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_area("Ask anything about Langchain")

if st.button("Submit"):
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    st.write(f"Response time: {time.process_time() - start_time}")
    st.write(response['answer'])