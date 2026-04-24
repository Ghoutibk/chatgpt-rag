import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable : {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier n'est pas un PDF : {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("Aucun contenu extrait du PDF (PDF vide ou scanné ?).")

    return documents


def clean_text(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return [doc for doc in documents if doc.page_content.strip()]


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,    
        keep_separator=False,    
    )
 
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


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


def chunk_statistics(chunks: list[Document]) -> dict:
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
