"""
================================================================================
 MODULE : rag_chain.py
 RÔLE   : Orchestrer le pipeline RAG complet — de la question utilisateur
          à la réponse finale générée par le LLM.
================================================================================

Ce module est le TROISIÈME MAILLON. Il met ensemble toutes les pièces :

    Question  ──▶  Recherche (vector_store)  ──▶  Prompt  ──▶  LLM  ──▶  Réponse
                          (les k meilleurs                   (Mistral)
                          chunks)


─── QU'EST-CE QU'UN LLM ? ──────────────────────────────────────────────────────

LLM = "Large Language Model" = modèle de langage de grande taille.

Concrètement, c'est un programme entraîné sur des milliards de textes,
capable de GÉNÉRER du texte en réponse à ce qu'on lui donne. ChatGPT,
Claude, Mistral-Large sont des LLM.

Ils ont deux gros défauts :
    1. Leur connaissance s'arrête à leur date d'entraînement (ex: 2024).
    2. Ils HALLUCINENT : ils inventent parfois des informations plausibles
       mais fausses, sans prévenir.

Le RAG sert précisément à CORRIGER ces deux défauts en leur fournissant
le contexte exact dont ils ont besoin pour répondre.


─── POURQUOI UN "PROMPT" AVEC DES RÈGLES ? ─────────────────────────────────────

Le "prompt" est le texte d'instruction qu'on donne au LLM avant qu'il
génère sa réponse. C'est la ZONE STRATÉGIQUE du projet : bien le rédiger,
c'est forcer le LLM à :

    • utiliser UNIQUEMENT les extraits du PDF (pas sa propre mémoire),
    • dire "je ne sais pas" si l'info n'est pas dans les extraits,
    • citer le document plutôt que de paraphraser de façon floue.

C'est là qu'on combat les hallucinations. Une mauvaise formulation du
prompt = un LLM qui dévie. On appelle cette discipline "prompt engineering".


─── POURQUOI MISTRAL POUR LE LLM AUSSI ? ───────────────────────────────────────

Pour ne pas multiplier les clés API et pour rester cohérent. On utilise :
    • "mistral-embed"        → pour les embeddings (vector_store.py)
    • "mistral-small-latest" → pour la génération (ce module)

"mistral-small" est un bon compromis : rapide, pas cher, largement
suffisant pour répondre à partir d'extraits. Pour un projet plus exigeant,
on pourrait passer à "mistral-large-latest".

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
#
# - "os" : pour lire la clé API depuis les variables d'environnement.
#
# - "ChatMistralAI" (de langchain_mistralai) : le client LangChain pour
#   l'API de GÉNÉRATION de Mistral. À ne pas confondre avec MistralAIEmbeddings
#   qui est pour la VECTORISATION.
#
# - "FAISS" : on l'utilise ici juste pour typer les paramètres de fonctions.
#
# - "PromptTemplate" (de langchain_core.prompts) : un outil pour créer des
#   prompts avec des variables (des "placeholders") qu'on remplira plus tard.
#   Exemple : "Résume ce texte : {texte}" avec {texte} remplaçable.
#
# - "Document" : toujours la structure de base LangChain.
# ============================================================================

import os
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ============================================================================
# LE PROMPT SYSTÈME — CŒUR ANTI-HALLUCINATION
# ============================================================================
#
# Ce template est ce qui sera envoyé au LLM à chaque question. Il est
# rédigé pour être STRICT : on ne laisse pas le LLM s'évader du document.
#
# Les règles numérotées sont délibérément répétitives et insistantes :
# les LLM ont tendance à suivre des instructions impératives claires.
# ============================================================================

RAG_PROMPT_TEMPLATE = """Tu es un assistant expert qui répond à des questions à partir d'un document fourni.

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les informations présentes dans le contexte ci-dessous.
2. Si la réponse ne se trouve pas dans le contexte, dis clairement : "Je ne trouve pas cette information dans le document."
3. N'invente jamais de données, chiffres, noms ou citations.
4. Cite les passages du document quand c'est pertinent.
5. Réponds de manière claire, concise et structurée.

CONTEXTE (extraits du document) :
{context}

QUESTION : {question}

RÉPONSE :"""


# ============================================================================
# INITIALISATION DU LLM
# ============================================================================

def get_llm(model: str = "mistral-small-latest", temperature: float = 0.1) -> ChatMistralAI:
    """
    Initialise le client du modèle de génération Mistral.

    Paramètre "temperature" (entre 0.0 et 1.0) :
        Contrôle le degré d'aléatoire dans les réponses du LLM.
            0.0 → déterministe, toujours la même réponse, factuel, "sec"
            0.7 → créatif, varié, adapté pour écrire une histoire
            1.0 → très aléatoire, parfois incohérent

        Pour un RAG factuel comme le nôtre, on veut peu de créativité
        et beaucoup de fidélité au texte source → 0.1 est idéal.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY non définie.")

    return ChatMistralAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


# ============================================================================
# FORMATAGE DES CHUNKS RÉCUPÉRÉS EN "CONTEXTE"
# ============================================================================

def format_docs(docs: list[Document]) -> str:
    """
    Prend la liste de chunks récupérés par la recherche, et les assemble
    en un unique gros bloc de texte numéroté, qu'on insérera dans le prompt.

    Exemple de sortie :

        [Extrait 1 - page 3]
        Le RAG combine une étape de recherche avec...

        [Extrait 2 - page 7]
        Les embeddings sont des vecteurs de haute dimension...

    La numérotation et l'indication de page servent deux buts :
        1. Aider le LLM à structurer sa réponse en citant les extraits.
        2. Permettre à l'utilisateur final de remonter à la source.
    """
    return "\n\n".join(
        f"[Extrait {i + 1} - page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ============================================================================
# FONCTION PRINCIPALE : RÉPONDRE À UNE QUESTION
# ============================================================================

def answer_with_sources(vector_store: FAISS, question: str, k: int = 4) -> dict:
    """
    Pipeline RAG complet pour UNE question utilisateur.

    Étapes internes :
        1. On transforme le vector_store en "retriever" (objet de recherche).
        2. On récupère les `k` chunks les plus similaires à la question.
        3. On formate ces chunks en un bloc de contexte.
        4. On construit le prompt final (template + contexte + question).
        5. On l'envoie au LLM Mistral.
        6. On renvoie la réponse ET les chunks sources (pour affichage UI).

    Retourne un dictionnaire :
        {
            "answer":  la réponse du LLM (texte),
            "sources": la liste des Document utilisés
                       (pour afficher les extraits dans l'interface)
        }
    """
    # 1. Le retriever : objet qui sait interroger la base vectorielle.
    #    Le paramètre k=... règle combien de chunks seront récupérés.
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 2. Recherche des chunks pertinents.
    #    .invoke(question) est la méthode LangChain moderne (anciennement
    #    .get_relevant_documents()).
    retrieved_docs = retriever.invoke(question)

    # 3. Construction du prompt final.
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    context = format_docs(retrieved_docs)
    formatted_prompt = prompt.format(context=context, question=question)

    # 4. Appel du LLM. .invoke() renvoie un AIMessage ; on récupère le
    #    texte via l'attribut .content.
    llm = get_llm()
    answer = llm.invoke(formatted_prompt).content

    # 5. On renvoie tout : la réponse et les sources.
    return {
        "answer": answer,
        "sources": retrieved_docs,
    }
