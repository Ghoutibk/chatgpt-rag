# ChatPDF — Assistant conversationnel basé sur RAG

Projet académique : implémenter un "ChatPDF" qui permet de poser des questions
à un document PDF, sans hallucination, grâce à une architecture RAG
(Retrieval-Augmented Generation).

## Architecture

```
PDF → Extraction texte → Chunking → Embeddings (Mistral) → FAISS
                                                              ↓
Question utilisateur → Embedding → Recherche similarité → Top-k chunks
                                                              ↓
                                      LLM (Mistral) + Prompt contraint → Réponse
```

## Structure du projet

```
chatpdf-rag/
├── app.py              # Interface Streamlit
├── pdf_processor.py    # Chargement et découpage des PDF
├── vector_store.py     # Embeddings Mistral + base FAISS
├── rag_chain.py        # Pipeline RAG (retrieval + génération)
├── requirements.txt    # Dépendances
├── .env.example        # Template pour la clé API
└── README.md           # Ce fichier (installation rapide)
```

## Installation

1. Se placer dans le dossier du projet :
   ```bash
   cd chatpdf-rag
   ```

2. Créer et activer un environnement virtuel Python (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate         # Linux / Mac
   venv\Scripts\activate            # Windows
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Configurer la clé API Mistral :
   ```bash
   cp .env.example .env
   ```
   Ouvrir le fichier `.env` et remplacer `votre_cle_api_ici` par votre
   vraie clé (obtenue sur https://console.mistral.ai/).

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse http://localhost:8501.

## Utilisation

1. Uploadez un PDF dans la barre latérale.
2. Ajustez si besoin les paramètres (taille des chunks, chevauchement, k).
3. Cliquez sur **Indexer le document**.
4. Posez vos questions dans la zone de chat. Les sources utilisées
   s'affichent sous chaque réponse.

## Modèles utilisés

- **Embeddings :** `mistral-embed` (vecteurs de dimension 1024)
- **LLM :** `mistral-small-latest` (modifiable dans `rag_chain.py`)
# chatgpt-rag
