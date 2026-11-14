import streamlit as st
import random
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA = os.environ.get("OLLAMA_URL")
LLM = os.environ.get("LLM_MODEL")

def get_qwen_response(messages):
    response = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": LLM, 
            "messages": messages,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response


st.title("Data quality assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Waiting for assistant..."):
            response = get_qwen_response(st.session_state.messages)
        data = response.json()
        reply = data["message"]["content"]

        def secondes(champ):
            return data.get(champ, 0) / 1_000_000_000

        st.caption(
            f"Chargement du modèle : {secondes('load_duration'):.1f} s · "
            f"Lecture du prompt : {secondes('prompt_eval_duration'):.1f} s · "
            f"Génération : {secondes('eval_duration'):.1f} s "
            f"({data.get('eval_count', 0)} tokens)"
        )
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

