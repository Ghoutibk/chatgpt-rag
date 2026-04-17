"""
================================================================================
 MODULE : vector_store.py
 RÔLE   : Transformer les chunks de texte en "vecteurs" et les ranger dans une
          base qui permet de retrouver les plus pertinents pour une question.
================================================================================

Ce module est le DEUXIÈME MAILLON du pipeline. Il gère deux concepts cruciaux :

    1) LES EMBEDDINGS (plongements vectoriels)
    2) LA BASE VECTORIELLE (ici FAISS)


─── QU'EST-CE QU'UN EMBEDDING ? ────────────────────────────────────────────────

Un embedding est une liste de nombres (typiquement 1024 nombres pour Mistral)
qui représente le SENS d'un texte.

    "Le chat mange" ──▶ [0.23, -0.45, 0.78, ..., 0.11]   (1024 nombres)

L'idée géniale : deux textes au SENS PROCHE auront des vecteurs PROCHES
dans cet espace à 1024 dimensions.

    "Le chat mange une souris"     ──▶ vecteur A
    "Le félin dévore un rongeur"   ──▶ vecteur B (très proche de A)
    "Il fait beau aujourd'hui"     ──▶ vecteur C (très éloigné de A)

On mesure la "proximité" entre deux vecteurs avec la SIMILARITÉ COSINUS.
Peu importe les détails mathématiques : retenez que cela donne un score
entre -1 et 1. Plus c'est proche de 1, plus les textes sont similaires
en sens.

Grâce à ça, quand l'utilisateur posera une question, on pourra calculer
son embedding, comparer à ceux des chunks, et retrouver les chunks les
plus proches EN SENS (pas seulement en mots-clés communs).


─── QU'EST-CE QU'UNE BASE VECTORIELLE ? ────────────────────────────────────────

C'est une structure de données spécialisée dans le stockage et la
recherche rapide de vecteurs. Sans elle, il faudrait comparer la question
à CHAQUE chunk un par un — lent si vous avez des milliers de chunks.

FAISS (développé par Meta/Facebook AI) utilise des algorithmes d'indexation
astucieux pour trouver les "k vecteurs les plus proches" en quelques
millisecondes, même sur des millions de vecteurs.


─── POURQUOI MISTRAL POUR LES EMBEDDINGS ? ─────────────────────────────────────

On a plusieurs options possibles :
    • Modèle local Hugging Face (sentence-transformers) : gratuit, hors ligne,
      mais demande de télécharger un modèle (~500 Mo) et un peu de puissance.
    • OpenAI (text-embedding-3-small/large) : performant mais payant et
      nécessite une clé OpenAI.
    • Mistral (mistral-embed) : excellent support du français, API simple,
      généreux en quota gratuit. Parfait pour un projet étudiant.

Mistral est un choix pertinent ici car :
    • Le projet est rédigé en français → Mistral est entraîné aussi sur du
      français, donc les embeddings sont de bonne qualité en français.
    • Une seule clé API suffit pour les embeddings ET le LLM (Phase 4).
    • Pas besoin d'infrastructure : tout se passe via Internet.

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
#
# - "os" : module standard Python pour accéder aux variables d'environnement
#   (c'est là qu'on lira MISTRAL_API_KEY).
#
# - "MistralAIEmbeddings" (de langchain_mistralai) : l'adaptateur LangChain
#   qui sait parler à l'API d'embeddings de Mistral. On lui donne un texte,
#   il renvoie le vecteur correspondant.
#
# - "FAISS" (de langchain_community.vectorstores) : la classe LangChain
#   qui enveloppe la bibliothèque FAISS de Meta. Elle se charge de :
#       1) Appeler le modèle d'embeddings pour vectoriser nos chunks
#       2) Construire l'index FAISS pour la recherche rapide
#       3) Sauvegarder / recharger l'index sur disque si besoin
#
# - "Document" : la même structure que dans pdf_processor.py.
# ============================================================================

import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================================
# INITIALISATION DU MODÈLE D'EMBEDDINGS
# ============================================================================

def get_embeddings() -> MistralAIEmbeddings:
    """
    Initialise la connexion à l'API d'embeddings de Mistral.

    Ne renvoie PAS un vecteur : renvoie un OBJET qu'on pourra utiliser
    ensuite pour vectoriser n'importe quel texte.

    Comment obtenir une clé API Mistral ?
        1. Aller sur https://console.mistral.ai/
        2. Créer un compte (gratuit)
        3. Générer une clé API
        4. La mettre dans un fichier `.env` à la racine du projet :
               MISTRAL_API_KEY=votre_cle_ici
    """
    # On récupère la clé depuis les variables d'environnement.
    # "os.getenv" renvoie None si la variable n'existe pas — d'où la vérif.
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY non définie. "
            "Ajoutez-la dans un fichier .env ou exportez-la dans votre shell."
        )

    # Le modèle "mistral-embed" produit des vecteurs de dimension 1024.
    # (Chaque texte sera représenté par une liste de 1024 nombres.)
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key,
    )


# ============================================================================
# CONSTRUCTION DE LA BASE VECTORIELLE
# ============================================================================

def build_vector_store(chunks: list[Document]) -> FAISS:
    """
    Prend la liste de chunks (produite par pdf_processor.py) et construit
    une base vectorielle FAISS complète.

    Ce qui se passe concrètement :
        1. Pour chaque chunk, un appel à l'API Mistral est fait pour obtenir
           son vecteur d'embedding (de dimension 1024).
        2. Tous les vecteurs sont rangés dans une structure d'index FAISS
           optimisée pour la recherche par similarité cosinus.
        3. Les métadonnées (source, page, chunk_id) sont conservées et
           associées à chaque vecteur.

    ATTENTION : chaque chunk = 1 appel API Mistral. Sur un gros PDF de
    500 chunks, cela veut dire 500 requêtes. Mistral regroupe les requêtes
    automatiquement (batching), mais cela peut prendre quelques dizaines
    de secondes.
    """
    embeddings = get_embeddings()

    # FAISS.from_documents fait TOUT le travail en une seule ligne :
    #   - récupère le texte de chaque chunk
    #   - appelle le modèle d'embeddings pour chacun
    #   - construit l'index FAISS
    #   - rattache les métadonnées
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


# ============================================================================
# SAUVEGARDE / RECHARGEMENT SUR DISQUE
# ============================================================================
#
# Pourquoi sauvegarder ? Parce que construire l'index coûte du temps et
# des appels API. Si on veut réutiliser le même PDF plus tard, autant
# garder l'index tout fait sur disque.
# ============================================================================

def save_vector_store(vector_store: FAISS, path: str = "faiss_index") -> None:
    """
    Écrit l'index FAISS dans un dossier sur disque.

    Ça crée un dossier "faiss_index/" contenant deux fichiers :
        - index.faiss  : les vecteurs indexés (fichier binaire)
        - index.pkl    : les métadonnées et les textes (fichier Python pickle)
    """
    vector_store.save_local(path)


def load_vector_store(path: str = "faiss_index") -> FAISS:
    """
    Recharge un index FAISS précédemment sauvegardé.

    Note : le paramètre `allow_dangerous_deserialization=True` est nécessaire
    car FAISS utilise le format "pickle" de Python, qui peut théoriquement
    contenir du code malveillant si on charge un fichier d'origine inconnue.
    Dans notre cas, on ne charge que NOS propres fichiers : c'est sûr.
    """
    embeddings = get_embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ============================================================================
# RECHERCHE PAR SIMILARITÉ (le cœur de la phase "Retrieval" du RAG)
# ============================================================================

def search(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    """
    Recherche les `k` chunks les plus similaires à la question `query`.

    Étapes internes (automatiques) :
        1. La question est vectorisée via l'API Mistral (1 appel, rapide).
        2. FAISS compare ce vecteur à tous les vecteurs de l'index.
        3. Les k chunks au score de similarité le plus élevé sont renvoyés.

    Le paramètre `k` est crucial :
        - k trop petit (ex: 1) : on risque de rater le chunk pertinent
        - k trop grand (ex: 20) : on sature le LLM avec trop de contexte
        - k = 3 à 5 est un bon compromis pour la plupart des cas.
    """
    return vector_store.similarity_search(query, k=k)
