import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def get_embeddings() -> MistralAIEmbeddings:

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY non définie. "
            "Ajoutez-la dans un fichier .env ou exportez-la dans votre shell."
        )

    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key,
    )


def build_vector_store(chunks: list[Document]) -> FAISS:
    
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "faiss_index") -> None:
    vector_store.save_local(path)


def load_vector_store(path: str = "faiss_index") -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    return vector_store.similarity_search(query, k=k)
