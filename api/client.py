import requests
import streamlit as st

def get_response(input_text, endpoint):
    response = requests.post(f"http://localhost:8080/{endpoint}/invoke", json={'input':{'topic': input_text}})
    try:
        json_response = response.json()
        return json_response['output']
    except:
        return response.json()

st.title("Langchain Demo with LangServe APIs")
input_text1 = st.text_input("Write an essay on")
input_text2 = st.text_input("Write an poem on")

if input_text1:
    st.write(get_response(input_text1, "essay"))
if input_text2:
    st.write(get_response(input_text2, "poem"))
