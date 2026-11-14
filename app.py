import sys
from pathlib import Path
from time import perf_counter

import psycopg
import requests
import streamlit as st

# Réutiliser les modules existants du projet.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from db import db_connection
from index import index_markdown
from rag import answer_question


def display_message(message):
    with st.chat_message(message["role"]):
        for filename in message.get("files", []):
            st.caption(f"📄 {filename}")
        if message["content"]:
            st.markdown(message["content"])


st.title("Data quality assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    display_message(message)

submission = st.chat_input(
    "Pose une question ou joins une documentation .md",
    accept_file="multiple",
    file_type=["md"],
    submit_mode="disable",
)

if submission is not None:
    prompt = submission.text.strip()
    files = submission.files

    if prompt or files:
        # Les noms de pièces jointes sont destinés à l'interface, pas à l'API.
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
            if message["content"]
        ]
        user_message = {
            "role": "user",
            "content": prompt,
            "files": [file.name for file in files],
        }
        st.session_state.messages.append(user_message)
        display_message(user_message)

        with st.chat_message("assistant"):
            upload_results = []
            try:
                if files:
                    with st.spinner("Lecture des documents…"):
                        with db_connection() as connection:
                            # Tous les fichiers de cet envoi sont validés ensemble.
                            with connection.transaction():
                                for file in files:
                                    result = index_markdown(
                                        connection, file.name, file.getvalue()
                                    )
                                    upload_results.append(result)

                    for result in upload_results:
                        if result["already_indexed"]:
                            st.caption(f"Déjà disponible : {result['filename']}")
                        else:
                            st.caption(f"Document ajouté : {result['filename']}")

                if prompt:
                    start = perf_counter()
                    with st.spinner("Recherche et génération de la réponse…"):
                        with db_connection() as connection:
                            reply, passages, tokens = answer_question(
                                connection,
                                question=prompt,
                                history=history,
                            )
                    st.markdown(reply)
                    st.caption(
                        f"Durée : {perf_counter() - start:.1f} s · "
                        f"{tokens['tokens_entree']} tokens en entrée · "
                        f"{tokens['tokens_sortie']} tokens générés"
                    )
                    with st.expander("Passages consultés"):
                        for number, passage in enumerate(passages, start=1):
                            st.write(f"[{number}] {passage['url']}")
                            st.text(passage["text"])
                else:
                    # Un dépôt seul n'a pas besoin d'un appel au modèle de chat.
                    reply = "\n\n".join(
                        f"{result['filename']} est déjà indexé."
                        if result["already_indexed"]
                        else f"{result['filename']} a été ajouté à la documentation."
                        for result in upload_results
                    )
                    st.markdown(reply)
            except (ValueError, OSError, psycopg.Error, requests.RequestException):
                reply = (
                    "L’opération a échoué. Vérifie le fichier Markdown UTF-8, "
                    "la connexion PostgreSQL et la disponibilité d’Ollama."
                )
                st.error(reply)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
