"""
================================================================================
 MODULE : app.py
 RÔLE   : Interface utilisateur web (Streamlit) qui utilise les 3 modules
          précédents pour offrir une expérience "ChatPDF".
================================================================================

C'est la façade visible du projet. Elle utilise :
    • pdf_processor.py  → pour traiter le PDF uploadé
    • vector_store.py   → pour construire la base vectorielle
    • rag_chain.py      → pour répondre aux questions


─── POURQUOI STREAMLIT ? ───────────────────────────────────────────────────────

Streamlit est un framework Python qui transforme un simple script en
application web. On n'a besoin de connaître NI HTML, NI CSS, NI JavaScript.

Avantages :
    • Ultra rapide à développer : 50 lignes Python = une app fonctionnelle.
    • Très adapté aux démos de data science / IA.
    • Widgets prêts à l'emploi : upload de fichier, chat, sliders, etc.

Inconvénients :
    • Pas fait pour servir des milliers d'utilisateurs simultanément.
    • Moins flexible qu'un vrai framework web (Flask, Django, FastAPI).

Pour un projet étudiant avec soutenance, Streamlit est parfait :
démonstration visuelle immédiate sans perdre de temps sur le front-end.


─── COMMENT FONCTIONNE STREAMLIT ? (à savoir pour ne pas être surpris) ─────────

Streamlit RE-EXÉCUTE tout le script à chaque interaction de l'utilisateur
(clic de bouton, saisie de texte, etc.). Du coup, si on n'y prend garde,
on perdrait la base vectorielle à chaque interaction !

Solution : `st.session_state`, un dictionnaire spécial qui PERSISTE entre
les ré-exécutions. On y stocke tout ce qu'on veut garder (la base
vectorielle, l'historique de la conversation...).

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
#
# - "os"        : lecture de variables d'environnement (clé API).
# - "tempfile"  : création de fichiers temporaires. Nécessaire car
#                 Streamlit nous donne le PDF uploadé sous forme d'objet
#                 en mémoire, mais PyPDFLoader a besoin d'un vrai chemin
#                 sur disque. On écrit donc le PDF dans un fichier temporaire
#                 le temps de l'indexer.
# - "streamlit" : le framework d'interface (importé comme "st" par convention).
# - "dotenv.load_dotenv" : charge les variables d'environnement depuis un
#                 fichier .env (pour ne pas mettre la clé API en dur dans le code).
# ============================================================================

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import process_pdf
from vector_store import build_vector_store
from rag_chain import answer_with_sources


# Lit le fichier .env et charge MISTRAL_API_KEY dans os.environ.
load_dotenv()


# ============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ============================================================================

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


# ============================================================================
# INITIALISATION DU SESSION STATE
# ============================================================================
#
# st.session_state est un dictionnaire qui survit aux ré-exécutions du script.
# On y stocke trois choses :
#     - vector_store : la base vectorielle du PDF courant (ou None)
#     - chat_history : la liste des messages échangés
#     - pdf_name     : le nom du PDF actif (pour l'affichage)
#
# On teste avec "if ... not in st.session_state" pour ne les initialiser
# qu'au tout premier lancement.
# ============================================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ============================================================================
# BARRE LATÉRALE — UPLOAD PDF ET PARAMÈTRES
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Vérification immédiate : sans clé API, rien n'est possible.
    # "st.stop()" arrête l'exécution proprement et affiche le message.
    if not os.getenv("MISTRAL_API_KEY"):
        st.error(
            "⚠️ MISTRAL_API_KEY non définie. "
            "Créez un fichier .env à partir de .env.example."
        )
        st.stop()

    # Widget d'upload de fichier. Renvoie None tant que rien n'est uploadé.
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

    # Sliders pour ajuster les paramètres du RAG.
    # L'utilisateur peut expérimenter en temps réel.
    chunk_size = st.slider("Taille des chunks (caractères)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chevauchement entre chunks", 0, 500, 200, step=50)
    k = st.slider("Nombre de passages récupérés (k)", 1, 10, 4)

    # Bouton pour lancer l'indexation — seulement si un fichier est uploadé.
    if uploaded_file is not None and st.button("📚 Indexer le document", type="primary"):

        # "st.spinner" affiche un petit indicateur de chargement pendant
        # l'exécution du bloc indenté ci-dessous.
        with st.spinner("Traitement du PDF en cours..."):

            # 1) On écrit le PDF uploadé dans un fichier temporaire sur disque.
            #    NamedTemporaryFile crée un fichier avec un nom aléatoire
            #    dans le dossier temp du système.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # 2) Extraction + chunking du PDF.
                chunks = process_pdf(
                    tmp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.info(f"✅ {len(chunks)} chunks créés à partir du PDF.")

                # 3) Vectorisation + indexation FAISS.
                st.session_state.vector_store = build_vector_store(chunks)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []  # reset de la conversation
                st.success(f"✅ Document « {uploaded_file.name} » indexé avec succès !")

            except Exception as e:
                # Si quoi que ce soit plante, on affiche l'erreur à l'utilisateur.
                st.error(f"Erreur lors de l'indexation : {e}")

            finally:
                # 4) Nettoyage : on supprime le fichier temporaire.
                #    Ce bloc "finally" s'exécute MÊME en cas d'exception.
                os.unlink(tmp_path)

    # Affichage du document actif et bouton de reset.
    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown(f"**Document actif :** `{st.session_state.pdf_name}`")
        if st.button("🗑️ Réinitialiser"):
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.rerun()  # force le rafraîchissement de l'interface


# ============================================================================
# ZONE PRINCIPALE — LE CHAT
# ============================================================================

# Si aucun PDF n'est indexé, on affiche un message d'accueil et on s'arrête.
if st.session_state.vector_store is None:
    st.info("👈 Uploadez un PDF et cliquez sur « Indexer le document » pour commencer.")

else:
    # AFFICHAGE DE L'HISTORIQUE DE CONVERSATION
    # On parcourt tous les messages passés et on les ré-affiche à chaque
    # ré-exécution du script (rappel : Streamlit re-run tout à chaque interaction).
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

    # ZONE DE SAISIE UTILISATEUR (en bas de l'écran, style ChatGPT).
    # La variable "question" reçoit le texte saisi quand l'utilisateur
    # valide (Entrée). Sinon, elle est None et on ne fait rien.
    if question := st.chat_input("Posez votre question sur le document..."):

        # On ajoute la question à l'historique et on l'affiche.
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Génération de la réponse (avec spinner pour indiquer le travail).
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération de la réponse..."):
                try:
                    # Appel du pipeline RAG complet.
                    result = answer_with_sources(
                        st.session_state.vector_store,
                        question,
                        k=k,
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                    # Affichage de la réponse.
                    st.markdown(answer)

                    # Affichage des sources dans un expander.
                    with st.expander(f"📎 Sources ({len(sources)} extraits)"):
                        for i, src in enumerate(sources, 1):
                            page = src.metadata.get("page", "?")
                            st.markdown(f"**Extrait {i} — page {page}**")
                            st.text(src.page_content)
                            st.markdown("---")

                    # Ajout à l'historique pour le prochain rafraîchissement.
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Erreur : {e}")
