import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import process_pdf
from vector_store import build_vector_store
from rag_chain import answer_with_sources

load_dotenv()



# CONFIGURATION DE LA PAGE STREAMLIT
st.set_page_config(
    page_title="ChatPDF - RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 ChatPDF — Assistant conversationnel basé sur RAG")
st.caption(
    "Posez des questions sur votre document PDF. "
    "Le système répond uniquement à partir du contenu du document."
)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# BARRE LATÉRALE — UPLOAD PDF ET PARAMÈTRES
with st.sidebar:
    st.header("⚙️ Configuration")

    if not os.getenv("MISTRAL_API_KEY"):
        st.error(
            "MISTRAL_API_KEY non définie. "
            "Créez un fichier .env à partir de .env.example."
        )
        st.stop()

    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

    # Sliders pour ajuster les paramètres du RAG.
    chunk_size = st.slider("Taille des chunks (caractères)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chevauchement entre chunks", 0, 500, 200, step=50)
    k = st.slider("Nombre de passages récupérés (k)", 1, 10, 4)

    # Bouton pour lancer l'indexation seulement si un fichier est uploadé.
    if uploaded_file is not None and st.button("📚 Indexer le document", type="primary"):

        with st.spinner("Traitement du PDF en cours..."):

            # On écrit le PDF uploadé dans un fichier temporaire sur disque.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # Extraction + chunking du PDF.
                chunks = process_pdf(
                    tmp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.info(f" {len(chunks)} chunks créés à partir du PDF.")

                # Vectorisation + indexation FAISS.
                st.session_state.vector_store = build_vector_store(chunks)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []  # reset de la conversation
                st.success(f"✅ Document « {uploaded_file.name} » indexé avec succès !")

            except Exception as e:
                st.error(f"Erreur lors de l'indexation : {e}")

            finally:
                # Nettoyage : on supprime le fichier temporaire.
                os.unlink(tmp_path)


    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown(f"**Document actif :** `{st.session_state.pdf_name}`")
        if st.button("🗑️ Réinitialiser"):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.rerun()  # force le rafraîchissement de l'interface


# Si aucun PDF n'est indexé, on affiche un message d'accueil et on s'arrête.
if st.session_state.vector_store is None:
    st.info("👈 Uploadez un PDF et cliquez sur « Indexer le document » pour commencer.")

else:
 
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Si le message a des sources (les chunks utilisés), on les
            # affiche dans un expander (zone repliable).
            if msg.get("sources"):
                with st.expander(f"📎 Sources ({len(msg['sources'])} extraits)"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Extrait {i} — page {page}**")
                        st.text(src.page_content)
                        st.markdown("---")

  
    if question := st.chat_input("Posez votre question sur le document..."):

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse..."):
                try:
                    result = answer_with_sources(
                        st.session_state.vector_store,
                        question,
                        k=k,
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    with st.expander(f"📎 Sources ({len(sources)} extraits)"):
                        for i, src in enumerate(sources, 1):
                            page = src.metadata.get("page", "?")
                            st.markdown(f"**Extrait {i} — page {page}**")
                            st.text(src.page_content)
                            st.markdown("---")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Erreur : {e}")
