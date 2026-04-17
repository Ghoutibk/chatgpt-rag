"""
================================================================================
 MODULE : pdf_processor.py
 RÔLE   : Lire un fichier PDF et le préparer pour la suite du pipeline RAG.
================================================================================

Ce module est le PREMIER MAILLON de la chaîne. Son travail se résume en 3 étapes :

    PDF brut  ──▶  Texte extrait page par page  ──▶  Texte nettoyé  ──▶  Chunks

Un "chunk" est un petit morceau de texte (typiquement un paragraphe ou deux).
On découpe le document en chunks car :

    1) Les modèles d'IA ne peuvent pas traiter un document entier d'un coup
       (leur "mémoire" est limitée). Il faut leur donner du texte par petits bouts.

    2) Quand l'utilisateur posera une question, on cherchera LES chunks pertinents
       pour y répondre. Plus les chunks sont bien découpés, meilleure sera la
       réponse.

    3) Un chunk trop GROS = on dilue l'information, la recherche devient floue.
       Un chunk trop PETIT = on perd le contexte, la réponse devient incomplète.

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
#
# Chaque import est choisi pour une raison précise. Voici l'explication :
#
# - "re" (expressions régulières) : module standard Python pour nettoyer du
#   texte (supprimer des espaces multiples, détecter des motifs, etc.).
#
# - "pathlib.Path" : manière moderne de manipuler des chemins de fichiers
#   en Python. Plus robuste que les vieilles fonctions "os.path".
#
# - "PyPDFLoader" (de langchain_community) : l'outil qui ouvre un PDF et en
#   extrait le texte. Il utilise la bibliothèque "pypdf" en coulisses.
#   On le préfère à un usage direct de pypdf parce qu'il renvoie déjà le
#   texte encapsulé dans des objets "Document" (avec métadonnées : nom du
#   fichier, numéro de page) que le reste de LangChain sait utiliser.
#
# - "RecursiveCharacterTextSplitter" (de langchain_text_splitters) : l'outil
#   qui découpe un long texte en chunks. Il s'appelle "Recursive" car il
#   essaie plusieurs séparateurs successifs (paragraphes, puis phrases,
#   puis mots) pour couper au "bon endroit".
#
# - "Document" (de langchain_core.documents) : la structure de données
#   standard de LangChain. Un Document contient deux choses : du texte
#   (page_content) et des métadonnées (metadata : un dictionnaire libre).
# ============================================================================

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================================
# ÉTAPE 1 - CHARGEMENT DU PDF
# ============================================================================

def load_pdf(file_path: str) -> list[Document]:
    """
    Charge un fichier PDF et renvoie son contenu page par page.

    Retourne : une liste de "Document" LangChain. Un Document par page.
        - Document.page_content → le texte de la page
        - Document.metadata    → dict avec "source" (chemin du fichier)
                                  et "page" (numéro de la page, commence à 0)

    Pourquoi on sépare par page ?
        Parce que c'est une granularité naturelle. Ça permet aussi de dire
        plus tard à l'utilisateur "cette réponse vient de la page 7".

    Limites à connaître :
        - Ne fonctionne PAS sur les PDF scannés (images de pages).
          Pour ceux-là, il faudrait un outil d'OCR (reconnaissance optique
          de caractères), type Tesseract ou un service cloud.
        - L'extraction des tableaux est souvent bancale : un tableau à
          plusieurs colonnes ressort en vrac.
    """
    # On crée un objet Path pour manipuler proprement le chemin du fichier.
    path = Path(file_path)

    # VÉRIFICATIONS DE SÉCURITÉ : on s'assure que le fichier existe et
    # que c'est bien un PDF. Sinon on lève une erreur explicite.
    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable : {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier n'est pas un PDF : {file_path}")

    # Création du "loader" (chargeur) et extraction du texte.
    # PyPDFLoader.load() retourne directement la liste de Document.
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Si le PDF n'a rien donné (ex: PDF vide ou 100% scanné), on alerte.
    if not documents:
        raise ValueError("Aucun contenu extrait du PDF (PDF vide ou scanné ?).")

    return documents


# ============================================================================
# ÉTAPE 2 - NETTOYAGE DU TEXTE
# ============================================================================

def clean_text(text: str) -> str:
    """
    Nettoie le texte extrait d'un PDF.

    Quand on extrait du texte d'un PDF, on obtient souvent des bizarreries
    dues à la mise en page du document original :

        EXEMPLE DE PROBLÈME 1 : mot coupé en fin de ligne
            Dans le PDF : "informa-
                          tion"
            Après extraction brute : "informa-\ntion"
            Ce qu'on veut : "information"

        EXEMPLE DE PROBLÈME 2 : sauts de ligne parasites
            Dans le PDF (une phrase sur 2 lignes à cause de la largeur) :
                "Le RAG est une
                architecture intéressante."
            Après extraction brute : "Le RAG est une\narchitecture intéressante."
            Ce qu'on veut : "Le RAG est une architecture intéressante."

        EXEMPLE DE PROBLÈME 3 : espaces multiples
            "trop      d'espaces" → "trop d'espaces"

    Les "expressions régulières" (regex) du module "re" sont l'outil idéal
    pour ce genre de nettoyage. Chaque ligne ci-dessous résout UN problème.
    """

    # 1. Recoller les mots coupés par un tiret en fin de ligne.
    #    Explication de la regex r"(\w+)-\n(\w+)" :
    #       - (\w+)  : un ou plusieurs caractères de mot (lettres, chiffres)
    #       - -\n    : un tiret suivi d'un saut de ligne
    #       - (\w+)  : encore des caractères de mot
    #    Le remplacement r"\1\2" : on colle le groupe 1 + le groupe 2.
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 2. Remplacer les SAUTS DE LIGNE SIMPLES par un espace, mais
    #    CONSERVER les doubles sauts de ligne (qui séparent les paragraphes).
    #    On veut garder la notion de paragraphe car le splitter s'en servira
    #    pour couper proprement.
    #    Regex : "(?<!\n)\n(?!\n)"
    #       - (?<!\n)  : pas de \n juste avant (negative lookbehind)
    #       - \n       : un saut de ligne
    #       - (?!\n)   : pas de \n juste après (negative lookahead)
    #    → ne match que les sauts de ligne "isolés".
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 3. Normaliser les espaces multiples en un seul espace.
    #    [ \t]+ = un espace OU une tabulation, répété.
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Limiter les sauts de ligne multiples à deux consécutifs maximum.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # .strip() supprime les espaces et sauts de ligne en début/fin de chaîne.
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Applique le nettoyage `clean_text` à tous les documents chargés.

    On modifie les documents sur place (on réassigne page_content), puis on
    FILTRE les pages qui sont devenues vides après nettoyage (ex: une page
    de garde qui contenait juste une image et rien de textuel).
    """
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Le "if doc.page_content.strip()" exclut les pages vides.
    return [doc for doc in documents if doc.page_content.strip()]


# ============================================================================
# ÉTAPE 3 - DÉCOUPAGE EN CHUNKS
# ============================================================================

def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Découpe chaque document (= page) en chunks plus petits.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ POURQUOI "Recursive" ?                                              │
    │                                                                     │
    │ Le splitter essaie de couper le texte selon une LISTE ORDONNÉE de   │
    │ séparateurs. Il prend le PREMIER qui permet de respecter la taille. │
    │                                                                     │
    │     1. "\n\n"  (entre paragraphes)   ← l'idéal                      │
    │     2. "\n"    (entre lignes)                                       │
    │     3. ". "    (entre phrases)                                      │
    │     4. " "     (entre mots)                                         │
    │     5. ""      (caractère par caractère) ← dernier recours          │
    │                                                                     │
    │ Résultat : les chunks respectent au maximum la structure naturelle  │
    │ du texte. On évite de couper au milieu d'un mot ou d'une phrase.    │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ POURQUOI un chevauchement (overlap) ?                               │
    │                                                                     │
    │ Imaginez une idée importante qui se trouve à cheval sur la fin      │
    │ d'un chunk et le début du suivant. Si on ne met pas d'overlap,      │
    │ aucun chunk ne contiendra l'idée complète, donc aucune recherche    │
    │ par similarité ne la trouvera efficacement.                         │
    │                                                                     │
    │ Avec un overlap de 200 caractères, les 200 derniers caractères      │
    │ d'un chunk sont aussi les 200 premiers du chunk suivant. Une idée   │
    │ importante a donc beaucoup plus de chances d'être capturée          │
    │ entièrement dans au moins un chunk.                                 │
    │                                                                     │
    │ Règle empirique : overlap ≈ 15 à 20 % de chunk_size.                │
    └─────────────────────────────────────────────────────────────────────┘

    Paramètres :
        chunk_size    : taille maximale d'un chunk, en CARACTÈRES
                        (1000 = ~150-200 mots = un paragraphe dense)
        chunk_overlap : nombre de caractères en commun entre deux chunks
                        consécutifs (200 = ~20% du chunk_size)
    """

    # Sécurité : l'overlap doit être strictement plus petit que la taille.
    # Sinon on tournerait en rond.
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")

    # Création du splitter avec notre configuration.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,      # on compte en caractères (pas en tokens)
        keep_separator=False,     # on ne garde pas les séparateurs eux-mêmes
    )

    # Le splitter travaille sur la liste de Document, pas sur des strings
    # brutes. Il préserve automatiquement les métadonnées de chaque page
    # (source, numéro de page) dans chaque chunk issu de cette page.
    chunks = splitter.split_documents(documents)

    # On enrichit les métadonnées de chaque chunk avec :
    #   - chunk_id  : un numéro unique et global pour le debug / traçage
    #   - chunk_size : la taille réelle du chunk (utile pour les stats)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


# ============================================================================
# ÉTAPE 4 - PIPELINE COMPLET (la fonction à appeler depuis l'extérieur)
# ============================================================================

def process_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clean: bool = True,
) -> list[Document]:
    """
    Pipeline complet : chargement → nettoyage (optionnel) → chunking.

    C'est la fonction qu'on appellera depuis l'interface Streamlit.
    Elle enchaîne les trois étapes précédentes.
    """
    documents = load_pdf(file_path)

    if clean:
        documents = clean_documents(documents)

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    return chunks


# ============================================================================
# ÉTAPE 5 - STATISTIQUES (utiles pour le rapport et le tuning)
# ============================================================================

def chunk_statistics(chunks: list[Document]) -> dict:
    """
    Calcule des statistiques sur une liste de chunks.

    Utile pour :
        - Comparer l'effet de différents chunk_size / chunk_overlap
        - Vérifier que le découpage produit des chunks cohérents
        - Remplir un tableau comparatif dans le rapport de projet
    """
    if not chunks:
        return {"nb_chunks": 0}

    sizes = [len(c.page_content) for c in chunks]
    pages = [c.metadata.get("page") for c in chunks if "page" in c.metadata]

    return {
        "nb_chunks": len(chunks),
        "taille_min": min(sizes),
        "taille_max": max(sizes),
        "taille_moy": round(sum(sizes) / len(sizes), 1),
        "taille_totale": sum(sizes),
        "nb_pages_couvertes": len(set(pages)) if pages else None,
    }
