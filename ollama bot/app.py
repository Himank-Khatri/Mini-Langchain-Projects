from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout='wide')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful coding assistant. Please respond to user queries. Do not write comments in the code. Return code in proper markdown but do not write too big font. Be concise and short, only show necessary information"),
        ("user", "Question: {question}")
    ]
)

with st.sidebar:
    model_name = st.selectbox("Model", ["llama3.2:1b", "llama3.2:3b", "deepseek-coder:1.3b", "deepseek-r1:1.5b", "qwen2.5:0.5b"])

llm = OllamaLLM(model=model_name)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your queries here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("processing..."):
        response = chain.invoke({'question': prompt})

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)