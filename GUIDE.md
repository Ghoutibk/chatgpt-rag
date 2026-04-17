# GUIDE COMPLET — ChatPDF avec architecture RAG

> Ce guide est rédigé pour être **accessible à une personne qui n'a
> jamais fait d'informatique**. Chaque concept est expliqué avec des
> analogies concrètes, et chaque choix technique est justifié.

---

## Table des matières

1. [Le problème qu'on veut résoudre](#1-le-problème-quon-veut-résoudre)
2. [Les concepts fondamentaux, vulgarisés](#2-les-concepts-fondamentaux-vulgarisés)
    - 2.1 [Qu'est-ce qu'un LLM ?](#21-quest-ce-quun-llm)
    - 2.2 [Le problème des hallucinations](#22-le-problème-des-hallucinations)
    - 2.3 [Qu'est-ce qu'une API ?](#23-quest-ce-quune-api)
    - 2.4 [Qu'est-ce qu'un embedding ?](#24-quest-ce-quun-embedding)
    - 2.5 [Qu'est-ce qu'une base vectorielle ?](#25-quest-ce-quune-base-vectorielle)
    - 2.6 [Qu'est-ce qu'un RAG ?](#26-quest-ce-quun-rag)
3. [Les choix techniques et leurs raisons](#3-les-choix-techniques-et-leurs-raisons)
    - 3.1 [Pourquoi Python ?](#31-pourquoi-python)
    - 3.2 [Pourquoi Mistral ?](#32-pourquoi-mistral)
    - 3.3 [Pourquoi LangChain ?](#33-pourquoi-langchain)
    - 3.4 [Pourquoi FAISS ?](#34-pourquoi-faiss)
    - 3.5 [Pourquoi PyPDF ?](#35-pourquoi-pypdf)
    - 3.6 [Pourquoi Streamlit ?](#36-pourquoi-streamlit)
4. [Architecture globale du projet](#4-architecture-globale-du-projet)
5. [Parcours d'une question, étape par étape](#5-parcours-dune-question-étape-par-étape)
6. [Explication détaillée de chaque fichier](#6-explication-détaillée-de-chaque-fichier)
    - 6.1 [`pdf_processor.py`](#61-pdf_processorpy)
    - 6.2 [`vector_store.py`](#62-vector_storepy)
    - 6.3 [`rag_chain.py`](#63-rag_chainpy)
    - 6.4 [`app.py`](#64-apppy)
7. [Exemple concret de bout en bout](#7-exemple-concret-de-bout-en-bout)
8. [Pistes d'amélioration](#8-pistes-daméliloration)
9. [Glossaire](#9-glossaire)

---

## 1. Le problème qu'on veut résoudre

Vous avez un document PDF — un contrat de 40 pages, un article scientifique
dense, le manuel technique d'un produit. Vous avez une question précise :
*« Quelle est la clause de résiliation ? »*, *« Quels sont les résultats
expérimentaux ? »*, *« Comment configurer le module X ? »*

**Aujourd'hui, vous avez deux options mauvaises :**

- **Lire tout le document** → long, fatigant, souvent inefficace.
- **Demander à ChatGPT** → risque de réponse inventée : ChatGPT n'a pas
  accès à VOTRE document et va peut-être confondre avec un autre qu'il a
  vu pendant son entraînement.

**Notre projet propose une troisième voie :** un assistant qui lit le PDF
POUR vous, et qui répond à vos questions UNIQUEMENT à partir de ce
document-là, en citant les passages qu'il a utilisés.

C'est exactement ce qu'on appelle un système **RAG** (Retrieval-Augmented
Generation). On va voir ce que ça veut dire, étape par étape.

---

## 2. Les concepts fondamentaux, vulgarisés

### 2.1 Qu'est-ce qu'un LLM ?

**LLM** = **Large Language Model** = modèle de langage de grande taille.

Un LLM est un programme informatique entraîné à partir d'énormes quantités
de texte (des milliards de pages web, livres, articles). À force de voir
du texte, il a appris à **compléter, reformuler, résumer, et générer du
texte** de manière très convaincante.

**Exemples de LLM que vous connaissez peut-être :**
- **ChatGPT** (OpenAI)
- **Claude** (Anthropic)
- **Mistral** (Mistral AI, une startup française)
- **Gemini** (Google)
- **Llama** (Meta, modèle libre)

**Analogie :** Un LLM, c'est comme un bibliothécaire très cultivé qui a
lu des millions de livres. Vous lui posez une question, et il vous
répond en utilisant tout ce qu'il a lu. Mais il a deux défauts majeurs :

1. Sa mémoire s'arrête à une certaine date (celle de son entraînement).
   Il ignore les événements récents.
2. Quand il ne sait pas, il **invente parfois au lieu d'avouer son
   ignorance**. C'est le problème des hallucinations.

### 2.2 Le problème des hallucinations

Une **hallucination** est une réponse d'un LLM qui a l'air correcte, est
bien formulée, mais qui est **factuellement fausse**. Le LLM n'est pas
malhonnête : il ne sait juste pas qu'il se trompe.

**Exemple réel observé chez ChatGPT :**

> *« Donne-moi la bibliographie de l'auteur X »*
>
> Réponse ChatGPT : une belle liste de 5 livres, dont 3 **n'existent pas**.
> Ils sont plausibles (bons titres, bonnes années, bon style), mais
> complètement inventés.

C'est évidemment inacceptable dans un contexte professionnel : contrat
juridique, diagnostic médical, décision financière, etc.

**Le RAG est la solution la plus utilisée pour limiter ce problème.**
L'idée : on FORCE le LLM à ne répondre qu'à partir d'un texte qu'on lui
fournit, plutôt que de le laisser puiser dans sa mémoire globale.

### 2.3 Qu'est-ce qu'une API ?

**API** = **Application Programming Interface** = interface de
programmation. Autrement dit : **un moyen pour un programme de parler à
un autre programme.**

**Analogie du restaurant :**
Vous êtes au restaurant. Vous voulez manger, mais vous n'allez pas
directement en cuisine. Vous parlez au **serveur**, qui transmet votre
commande à la cuisine, puis vous ramène votre plat.

Dans cette analogie :
- Vous = votre programme (notre ChatPDF)
- La cuisine = le service qui fait le travail (Mistral qui héberge
  l'intelligence artificielle)
- Le serveur = l'API

Concrètement, dans notre projet, on **envoie via Internet** un petit
message à Mistral (« voici un texte, donne-moi son embedding »), et
Mistral nous **renvoie le résultat** (« voici le vecteur correspondant »).
Cela nécessite :

1. Une **clé API** (comme un badge d'accès, qui identifie votre compte).
2. Une connexion Internet.

**Avantage d'une API :** on n'a pas besoin d'héberger soi-même le modèle
d'IA (qui pèserait plusieurs gigaoctets et demanderait une carte
graphique puissante). On délègue le travail à Mistral.

**Inconvénient :** on dépend d'un service externe (Internet, quotas,
confidentialité des données envoyées).

### 2.4 Qu'est-ce qu'un embedding ?

C'est **LE concept central** du RAG. Prenez le temps de bien le
comprendre.

**Un embedding est une liste de nombres qui représente le SENS d'un
texte.**

Plus précisément : un modèle d'embedding prend un texte en entrée
(quelques mots, une phrase, un paragraphe) et renvoie une liste de
plusieurs centaines de nombres (chez Mistral : 1024 nombres).

```
"Le chat mange une souris"
        │
        ▼ (modèle d'embedding)
        │
[0.23, -0.45, 0.78, 0.12, ..., 0.11]   ← 1024 nombres
```

**La magie, c'est que deux textes au SENS PROCHE produisent des listes
de nombres PROCHES.**

```
"Le chat mange une souris"    ──▶ [0.23, -0.45, 0.78, ...]   ← Vecteur A
"Le félin dévore un rongeur"  ──▶ [0.25, -0.42, 0.80, ...]   ← Vecteur B (très proche de A)
"Il fait beau aujourd'hui"    ──▶ [-0.67, 0.88, -0.12, ...]  ← Vecteur C (très éloigné)
```

**Analogie géographique :**

Imaginez une carte de France. Paris et Versailles sont proches sur la
carte. Paris et Marseille sont loin. La « distance géographique » entre
deux villes reflète leur proximité physique.

Un embedding fait pareil, mais avec des **textes au lieu de villes**,
et sur une « carte » à **1024 dimensions au lieu de 2**. La distance
entre deux embeddings reflète leur proximité de **sens**.

**Pourquoi 1024 dimensions ?**

Parce qu'une seule dimension ne permet pas de capturer toute la richesse
sémantique du langage. Avec 2 dimensions (une carte), on peut dire
« proche » ou « loin ». Avec 1024 dimensions, on peut dire « proche sur
le sujet du temps, lointain sur le sujet des animaux, proche sur le
vocabulaire formel... » — bref, une distinction fine selon 1024 axes
indépendants.

**Pas besoin de comprendre les maths.** Retenez juste :
> *« L'embedding transforme du texte en une position sur une carte de
> sens. Les textes qui parlent de la même chose sont voisins sur cette
> carte. »*

### 2.5 Qu'est-ce qu'une base vectorielle ?

Maintenant qu'on sait transformer des textes en vecteurs, on a un problème
pratique : si on a **des milliers de chunks** (morceaux de texte), on
a des milliers de vecteurs. Quand l'utilisateur pose une question, il
faut comparer le vecteur de la question à **tous ces vecteurs** pour
trouver les plus proches.

Fait naïvement, ça voudrait dire comparer 1 vecteur à 10 000 vecteurs,
un par un : lent.

**Une base vectorielle est une structure spécialisée qui indexe les
vecteurs pour accélérer cette recherche.** Elle utilise des algorithmes
sophistiqués pour trouver les « k plus proches voisins » en quelques
millisecondes, même sur des millions de vecteurs.

**Analogie de la bibliothèque :**

Dans une bibliothèque classique, les livres sont rangés **par ordre
alphabétique** (d'auteur, de titre). Très pratique si vous cherchez
*« Le Petit Prince »*. Mais inutile si vous cherchez *« un livre sur
l'aviation poétique »* : il faudrait regarder chaque livre un par un.

Une base vectorielle, c'est une bibliothèque où les livres sont rangés
**par similarité de contenu** (tous les livres sur la guerre sont
proches, tous les livres de cuisine sont proches, etc.). Si vous
arrivez avec un livre et demandez *« trouve-moi les 5 plus similaires »*,
la bibliothèque vous les donne directement.

**FAISS (qu'on utilise dans ce projet)** est une bibliothèque développée
par Meta (Facebook) qui fait exactement ça, mais pour des vecteurs
numériques.

### 2.6 Qu'est-ce qu'un RAG ?

**RAG** = **Retrieval-Augmented Generation** = « Génération Augmentée
par Récupération ».

C'est une architecture qui combine deux étapes :

1. **Retrieval (récupération)** : quand l'utilisateur pose une question,
   on cherche dans notre base vectorielle les chunks de texte les plus
   pertinents pour cette question.

2. **Augmented Generation (génération augmentée)** : on donne ces chunks
   AU LLM avec la question, et on lui demande de rédiger une réponse
   en se basant UNIQUEMENT sur ces chunks.

**Pourquoi c'est génial ?**

- Le LLM ne puise plus dans sa mémoire globale (risque d'hallucination),
  mais dans le texte précis du document qu'on lui fournit.
- Si l'information n'est pas dans le document, un prompt bien rédigé
  force le LLM à le dire (*« je ne trouve pas cette information »*).
- On peut montrer à l'utilisateur **d'où vient la réponse** (les chunks
  utilisés), ce qui permet la vérification.

**Schéma du flux complet d'un RAG :**

```
  PHASE 1 : PRÉPARATION (une fois par document)
  ──────────────────────────────────────────────────────────────────
  PDF → extraction texte → découpage en chunks → embeddings → FAISS

  PHASE 2 : INTERROGATION (à chaque question)
  ──────────────────────────────────────────────────────────────────
  Question → embedding → FAISS trouve les k chunks proches
             ↓
         Prompt (règles + chunks + question)
             ↓
          LLM génère la réponse
             ↓
       Réponse + sources
```

---

## 3. Les choix techniques et leurs raisons

Pour chaque composant du projet, on a eu plusieurs options. Voici les
choix faits et pourquoi.

### 3.1 Pourquoi Python ?

**Python est le langage standard en IA et science des données**, pour
plusieurs raisons :

- **Toutes les grandes bibliothèques d'IA sont en Python** (PyTorch,
  TensorFlow, Hugging Face, LangChain...). Autre langage = on rame.
- **Syntaxe simple et lisible** : on se concentre sur la logique plutôt
  que sur la technique du langage.
- **Écosystème riche** : il y a une bibliothèque pour à peu près tout.

**Alternatives écartées :**
- **JavaScript/TypeScript** : possible avec LangChain.js, mais
  l'écosystème IA est moins mature en JS.
- **Java, C#, Go** : rares pour faire de l'IA, très peu de ressources.
- **C++** : très performant mais trop bas niveau et chronophage pour
  un projet étudiant.

### 3.2 Pourquoi Mistral ?

On a trois grandes familles de choix pour les **embeddings** et le
**LLM** :

| Option | Avantages | Inconvénients |
|---|---|---|
| **Modèles locaux** (Hugging Face, Ollama) | Gratuit, données chez soi, pas besoin d'Internet | Installation lourde (~500 Mo-15 Go), carte graphique recommandée, qualité variable |
| **API OpenAI** (ChatGPT, text-embedding-3) | Qualité au sommet | Payant dès les premiers usages, clé US, problèmes RGPD potentiels |
| **API Mistral** | Excellent en français, startup française, quota gratuit généreux | Nécessite Internet, quota limité à terme |

**Pourquoi Mistral est particulièrement adapté à ce projet :**

1. **Support du français de premier ordre.** Mistral est une société
   française ; ses modèles sont entraînés avec beaucoup de français.
   Si vos PDF sont en français, les embeddings sont de meilleure
   qualité qu'avec des modèles anglophones comme ceux d'OpenAI.

2. **Une seule clé API pour les deux usages** (embeddings + génération).
   Simplifie énormément la configuration.

3. **Quota gratuit suffisant pour un projet étudiant.** On peut indexer
   et interroger plusieurs PDF sans jamais atteindre la limite.

4. **Qualité/coût excellent.** Les modèles Mistral rivalisent avec
   ceux d'OpenAI à prix plus bas.

5. **Souveraineté numérique.** Les données sont traitées en Europe (RGPD).

### 3.3 Pourquoi LangChain ?

**LangChain** est un framework qui fournit des "briques" prêtes à
l'emploi pour construire des applications autour des LLM. C'est le
chef d'orchestre qui fait parler ensemble :

- le chargeur de PDF,
- le découpeur de texte,
- le modèle d'embedding,
- la base vectorielle,
- le LLM,
- l'interface.

**Alternatives envisageables :**

| Option | Description | Verdict |
|---|---|---|
| **LangChain** | Framework le plus populaire, énormément de tutoriels | ✅ Choisi |
| **LlamaIndex** | Concurrent direct, particulièrement fort sur le RAG | ➕ Bon aussi, mais écosystème un peu plus étroit |
| **"From scratch"** (sans framework) | Coder tous les appels API à la main | ❌ Beaucoup plus long, pas d'intérêt pédagogique supplémentaire ici |
| **Haystack** | Framework de Deepset | ➖ Moins populaire, moins de tutos en français |

**Pourquoi LangChain plutôt que LlamaIndex ?**

Les deux se valent sur un projet RAG simple comme le nôtre. **LangChain
a une communauté plus large** (plus de tutoriels, Stack Overflow, etc.),
ce qui aide à débloquer les problèmes rencontrés. LlamaIndex est parfois
plus élégant pour des RAG complexes (multi-documents, agents), mais ces
subtilités dépassent le cadre de ce projet.

**Ce que LangChain nous apporte concrètement :**

- `PyPDFLoader` : charge un PDF en 1 ligne.
- `RecursiveCharacterTextSplitter` : découpe en chunks intelligemment.
- `MistralAIEmbeddings` : interface prête pour l'API d'embeddings Mistral.
- `FAISS` : wrapper autour de la bibliothèque FAISS de Meta.
- `ChatMistralAI` : interface prête pour l'API de génération Mistral.
- `PromptTemplate` : gestion propre des prompts avec variables.

Sans LangChain, il faudrait écrire toute la "plomberie" à la main :
plusieurs centaines de lignes supplémentaires sans valeur ajoutée.

### 3.4 Pourquoi FAISS ?

On a le choix entre plusieurs bases vectorielles :

| Option | Type | Points forts | Points faibles |
|---|---|---|---|
| **FAISS** | Bibliothèque (locale, en mémoire) | Très rapide, zéro config, créée par Meta | Pas de "vraie" base de données, pas multi-utilisateurs |
| **ChromaDB** | Base de données (fichier local ou serveur) | Interface plus "base de données", persistance native | Un peu plus lent, setup légèrement plus lourd |
| **Pinecone, Weaviate, Qdrant** | Services cloud / serveurs | Scalables à des millions de documents, production-ready | Inscription, parfois payants, complexes pour un projet étudiant |

**Pourquoi FAISS pour notre projet ?**

1. **Simplicité absolue.** Aucune configuration, aucun serveur à lancer.
   L'index FAISS tient dans la mémoire du programme pendant la session.

2. **Vitesse.** FAISS est codé en C++ (exposé en Python) et optimisé
   par Meta pour la recherche de vecteurs. C'est plus rapide que la
   plupart des alternatives sur des bases de petite à moyenne taille.

3. **Adapté à la taille du projet.** On traite un PDF à la fois, donc
   typiquement quelques centaines de chunks. FAISS en mémoire est
   parfait pour ça. Pas besoin d'une base distribuée.

4. **Sauvegarde/chargement faciles** sur disque (2 fichiers). Si on
   voulait persister l'index entre deux sessions, c'est trivial.

**Quand aurait-on préféré ChromaDB ?** Si le projet avait besoin de
gérer une bibliothèque croissante de documents partagés entre plusieurs
utilisateurs, ChromaDB (avec son mode serveur) serait plus adapté.

### 3.5 Pourquoi PyPDF ?

**PyPDF** est une bibliothèque Python pour lire les fichiers PDF
(extraire le texte, les métadonnées, parfois les images).

**Alternatives :**

| Bibliothèque | Avantages | Inconvénients |
|---|---|---|
| **PyPDF** | Simple, pur Python, pas de dépendances externes | Extraction parfois imparfaite (tableaux, colonnes) |
| **PyMuPDF (fitz)** | Extraction de meilleure qualité, rapide | Licence AGPL (contraignante pour certains usages commerciaux) |
| **pdfplumber** | Excellent pour les tableaux | Plus lent |
| **Tesseract / OCR** | Nécessaire pour les PDF scannés (images) | Setup complexe, plus lent |

**Pourquoi PyPDF dans notre projet ?**

1. **C'est ce que LangChain utilise par défaut** avec `PyPDFLoader`.
   Pas besoin d'installer autre chose.

2. **Simple et suffisant** pour la majorité des PDF "texte" (papiers
   scientifiques, contrats, rapports). Ce sont nos cibles principales.

3. **Licence permissive** (BSD) : pas de souci de réutilisation.

**Limite à connaître :** PyPDF ne gère pas les PDF scannés (images). Si
votre PDF est un vieux document numérisé, PyPDF renverra du vide ou du
texte incompréhensible. Il faudrait un outil d'OCR (type Tesseract)
pour y remédier — hors du périmètre de ce projet.

### 3.6 Pourquoi Streamlit ?

On a besoin d'une interface où l'utilisateur peut **uploader un PDF,
taper ses questions, et voir les réponses**. Plusieurs options :

| Framework | Description | Verdict |
|---|---|---|
| **Streamlit** | Transforme un script Python en app web en ~50 lignes | ✅ Choisi |
| **Gradio** | Très similaire à Streamlit, orienté démos ML | ➕ Bon aussi, un peu moins polyvalent |
| **Flask / FastAPI** | Frameworks web classiques, front-end à faire à la main | ❌ Trop de code front-end pour un projet IA |
| **Django** | Framework web complet | ❌ Disproportionné pour ce besoin |
| **React / Vue** | Framework JavaScript, front-end pur | ❌ Exige de coder un back-end à côté |

**Pourquoi Streamlit est parfait ici :**

1. **Pas besoin de HTML, CSS, JavaScript.** Tout est en Python.

2. **Widgets prêts à l'emploi** : `st.file_uploader`, `st.chat_input`,
   `st.chat_message`, `st.slider`... Tout ce qu'il nous faut.

3. **Démarrage instantané** : `streamlit run app.py` et on a une page
   web à http://localhost:8501.

4. **Parfait pour soutenance** : on partage l'écran, on montre l'appli
   en direct. Effet "wow" garanti.

**Limite à connaître :** Streamlit ré-exécute tout le script à chaque
interaction utilisateur. D'où l'usage de `st.session_state` pour
conserver la base vectorielle entre deux clics (sinon on la reconstruirait
à chaque message).

---

## 4. Architecture globale du projet

Le projet est découpé en **4 modules Python**, chacun avec une
responsabilité claire. Cette séparation est un principe de base de
l'ingénierie logicielle : **single responsibility principle** — chaque
module fait une chose, et la fait bien.

```
┌──────────────────────────────────────────────────────────────────┐
│                           app.py                                 │
│              (interface utilisateur - Streamlit)                 │
│                                                                  │
│  • Upload du PDF                                                 │
│  • Zone de chat                                                  │
│  • Affichage des réponses et des sources                         │
│  • Sliders de paramètres (chunk_size, k, etc.)                   │
└──────────────────────┬───────────────────────────────────────────┘
                       │ utilise
          ┌────────────┼────────────┐
          ▼            ▼            ▼
  ┌─────────────┐ ┌──────────┐ ┌──────────┐
  │pdf_processor│ │  vector_ │ │ rag_chain│
  │     .py     │ │ store.py │ │   .py    │
  │             │ │          │ │          │
  │ Lit le PDF, │ │ Calcule  │ │ Orchestre│
  │ nettoie le  │ │ les      │ │ la       │
  │ texte,      │ │ embed-   │ │ recherche│
  │ découpe en  │ │ dings    │ │ + la     │
  │ chunks.     │ │ (API     │ │ généra-  │
  │             │ │ Mistral) │ │ tion     │
  │ → utilise   │ │ + FAISS. │ │ (API     │
  │ PyPDFLoader │ │          │ │ Mistral).│
  │ et Splitter │ │          │ │          │
  └─────────────┘ └──────────┘ └──────────┘
         │              │             │
         ▼              ▼             ▼
  [ LangChain + PyPDF ] [ API Mistral + FAISS ] [ API Mistral ]
```

**Pourquoi cette séparation ?**

- **`pdf_processor.py` ne sait pas que Mistral existe.** Si un jour on
  change d'outil d'embeddings, on ne touche pas à ce module.
- **`vector_store.py` ne sait pas qu'on utilise Streamlit.** Si demain
  on passe à une autre interface, on ne touche qu'à `app.py`.
- **Chaque module est testable indépendamment.** On peut vérifier que
  le PDF est bien découpé sans avoir à lancer toute l'application.

---

## 5. Parcours d'une question, étape par étape

Voici ce qui se passe **dans l'ordre chronologique** quand un
utilisateur pose une question. C'est la manière la plus claire de
comprendre comment les morceaux s'assemblent.

### Phase A — Préparation (une fois par PDF)

**Étape 1.** L'utilisateur uploade un PDF via `app.py`.

**Étape 2.** `app.py` écrit le PDF dans un fichier temporaire sur disque
(car PyPDFLoader lit depuis un chemin, pas depuis la mémoire).

**Étape 3.** `app.py` appelle `process_pdf(...)` dans `pdf_processor.py`,
qui :
- charge le PDF avec `PyPDFLoader` → liste de Document (un par page),
- nettoie le texte (supprime les césures, sauts de ligne parasites, etc.),
- découpe en chunks avec `RecursiveCharacterTextSplitter` → liste finale
  de ~50-500 chunks selon la taille du PDF.

**Étape 4.** `app.py` appelle `build_vector_store(chunks)` dans
`vector_store.py`, qui :
- initialise `MistralAIEmbeddings` (connexion à l'API Mistral),
- envoie tous les chunks à Mistral pour obtenir leurs embeddings (par
  lots, automatiquement),
- range les embeddings dans un index FAISS en mémoire.

**Étape 5.** L'index FAISS est stocké dans `st.session_state` pour
survivre aux ré-exécutions Streamlit.

À la fin de cette phase, la base vectorielle est prête : on peut
maintenant répondre à des questions.

### Phase B — Interrogation (à chaque question)

**Étape 6.** L'utilisateur tape une question dans le chat.

**Étape 7.** `app.py` appelle `answer_with_sources(vector_store, question, k)`
dans `rag_chain.py`, qui :
- crée un "retriever" à partir du vector_store,
- envoie la question à l'API Mistral pour l'embeddinguer (1 appel rapide),
- demande à FAISS les `k` chunks dont les embeddings sont les plus
  proches → liste de chunks candidats,
- formate ces chunks en un bloc de contexte (avec numéros et pages),
- construit le prompt final en insérant le contexte et la question dans
  le template `RAG_PROMPT_TEMPLATE`,
- envoie le prompt à `ChatMistralAI` (le LLM),
- récupère la réponse textuelle.

**Étape 8.** `rag_chain.py` renvoie un dictionnaire `{"answer": ..., "sources": [...]}`.

**Étape 9.** `app.py` affiche la réponse et les sources (dans un
`st.expander` pour pouvoir les replier). Elle ajoute aussi le message
à l'historique `st.session_state.chat_history` pour qu'il reste visible
dans les prochains affichages.

---

## 6. Explication détaillée de chaque fichier

Chaque fichier du projet commence par un gros commentaire en en-tête qui
explique son rôle et les choix faits. Ce qui suit résume et complète.

### 6.1 `pdf_processor.py`

**Rôle :** lire un PDF et préparer ses données pour la suite.

**Fonctions principales :**

- **`load_pdf(file_path)`** — Ouvre le PDF, extrait une liste de
  Document (un par page) via `PyPDFLoader`. Vérifie que le fichier existe
  et est bien un PDF.

- **`clean_text(text)`** — Nettoie un bloc de texte : recolle les mots
  coupés en fin de ligne (`informa-\ntion` → `information`), remplace
  les sauts de ligne parasites par des espaces, normalise les espaces
  multiples. Les expressions régulières (`re.sub(...)`) sont
  l'outil parfait pour ce genre de traitement motif par motif.

- **`clean_documents(documents)`** — Applique `clean_text` à chaque
  page et retire les pages vides après nettoyage.

- **`split_documents(documents, chunk_size, chunk_overlap)`** — Utilise
  `RecursiveCharacterTextSplitter` pour découper chaque page en chunks.
  Le splitter essaie en priorité de couper aux paragraphes, puis aux
  lignes, puis aux phrases, puis aux mots — pour garder la cohérence
  sémantique.

- **`process_pdf(...)`** — Pipeline complet : chargement + nettoyage
  + découpage. C'est la seule fonction exposée vers `app.py`.

- **`chunk_statistics(chunks)`** — Statistiques descriptives sur une
  liste de chunks (utile pour le rapport et le tuning des paramètres).

**Paramètres clés :**
- `chunk_size` (1000 par défaut) : taille maximale d'un chunk en
  caractères.
- `chunk_overlap` (200 par défaut) : nombre de caractères partagés
  entre deux chunks consécutifs, pour éviter de couper une idée à
  cheval entre deux chunks.

### 6.2 `vector_store.py`

**Rôle :** transformer les chunks en embeddings (via l'API Mistral) et
les ranger dans un index FAISS pour la recherche par similarité.

**Fonctions principales :**

- **`get_embeddings()`** — Initialise l'objet `MistralAIEmbeddings`.
  Il n'est pas utilisé directement : on le passe ensuite à FAISS. Lit
  la clé API depuis la variable d'environnement `MISTRAL_API_KEY`.

- **`build_vector_store(chunks)`** — Appelle `FAISS.from_documents(...)`
  qui, en une seule ligne, calcule tous les embeddings (via l'API
  Mistral) et les indexe dans FAISS. Les métadonnées (source, page,
  chunk_id) sont conservées automatiquement.

- **`save_vector_store` / `load_vector_store`** — Sauvegarde et
  rechargement de l'index sur disque (pour ne pas refaire tous les
  appels API si on réutilise le même PDF plus tard).

- **`search(vector_store, query, k)`** — Renvoie les `k` chunks dont
  les embeddings sont les plus proches du vecteur de la question.

**Modèle utilisé :** `mistral-embed`. Il produit des vecteurs de
**dimension 1024**. La métrique de similarité par défaut dans FAISS
est la similarité cosinus, adaptée aux embeddings sémantiques modernes.

### 6.3 `rag_chain.py`

**Rôle :** orchestrer le pipeline RAG complet — recherche + génération.

**Fonctions principales :**

- **`get_llm(model, temperature)`** — Initialise `ChatMistralAI` avec
  le modèle `mistral-small-latest` et une température basse (0.1) pour
  des réponses factuelles et reproductibles.

- **`format_docs(docs)`** — Formate la liste de chunks récupérés en
  un unique bloc de texte numéroté, avec indication de page. Ce bloc
  sera inséré dans le prompt.

- **`answer_with_sources(vector_store, question, k)`** — Fonction
  centrale : fait la recherche, construit le prompt, appelle le LLM,
  renvoie la réponse + les sources.

**Élément stratégique : `RAG_PROMPT_TEMPLATE`**

Le prompt est la **ligne de défense anti-hallucination**. Sa formulation
est essentielle :

```
Tu es un assistant expert qui répond à des questions à partir d'un document fourni.

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les informations présentes dans le contexte ci-dessous.
2. Si la réponse ne se trouve pas dans le contexte, dis clairement :
   "Je ne trouve pas cette information dans le document."
3. N'invente jamais de données, chiffres, noms ou citations.
4. Cite les passages du document quand c'est pertinent.
5. Réponds de manière claire, concise et structurée.

CONTEXTE (extraits du document) :
{context}

QUESTION : {question}

RÉPONSE :
```

**Pourquoi cette rédaction ?**

- **Règles numérotées et impératives** : les LLM suivent mieux les
  instructions formulées comme des règles claires que des suggestions.
- **« UNIQUEMENT » en majuscules** : effet de saillance ; le LLM y
  prête plus attention.
- **Phrase exacte à dire en cas d'ignorance** : on réduit le risque
  de formulations vagues ou inventées.
- **« Cite les passages » encourage la traçabilité** de la réponse.

### 6.4 `app.py`

**Rôle :** interface web Streamlit qui met tout en musique.

**Structure :**

1. **Configuration de page** (`st.set_page_config`, `st.title`).
2. **Initialisation du session_state** (persistance entre rafraîchissements).
3. **Barre latérale** : vérification de la clé API, upload de PDF,
   sliders de paramètres, bouton d'indexation.
4. **Zone principale** : affichage de l'historique de chat, champ de
   saisie, appel du pipeline RAG, affichage de la réponse et des sources.

**Points techniques subtils :**

- **`with tempfile.NamedTemporaryFile(...)`** : crée un fichier
  temporaire sur disque pour y écrire le PDF uploadé. Nécessaire car
  `PyPDFLoader` veut un chemin, pas un objet en mémoire. On supprime le
  fichier temporaire après usage dans un bloc `finally` (même en cas
  d'erreur).

- **`st.session_state`** : dictionnaire spécial qui survit aux
  rafraîchissements. On y stocke le vector_store, l'historique des
  messages et le nom du PDF courant.

- **`st.spinner("...")`** : affiche un indicateur de chargement
  pendant une opération longue (indexation, génération de réponse).

- **`st.expander("Sources")`** : zone repliable qui montre les chunks
  utilisés pour la réponse — transparence pour l'utilisateur.

---

## 7. Exemple concret de bout en bout

Pour fixer les idées, déroulons un exemple fictif.

**Contexte :** on a uploadé un PDF de 10 pages sur l'architecture RAG.

**Étape 1 — Indexation**

Après extraction et nettoyage, le document fait ~20 000 caractères.
Avec `chunk_size=1000` et `chunk_overlap=200`, on obtient **25 chunks**.

On appelle l'API Mistral pour embedder les 25 chunks → on reçoit 25
vecteurs de 1024 nombres chacun. FAISS les indexe. Tout est prêt en
~15 secondes.

**Étape 2 — Question**

L'utilisateur tape : *« Qu'est-ce qu'un embedding ? »*

**Étape 3 — Retrieval**

1. L'API Mistral embedding la question → un vecteur de 1024 nombres.
2. FAISS compare ce vecteur aux 25 vecteurs indexés.
3. FAISS renvoie les 4 chunks au score cosinus le plus élevé.

Imaginons qu'elle renvoie :
- chunk A : extrait expliquant les embeddings (page 3) — score 0.89
- chunk B : extrait sur la dimension des vecteurs (page 4) — score 0.81
- chunk C : extrait sur la similarité cosinus (page 5) — score 0.74
- chunk D : extrait sur FAISS (page 6) — score 0.68

**Étape 4 — Construction du prompt**

Le bloc de contexte assemblé :

```
[Extrait 1 - page 3]
Un embedding est une représentation vectorielle d'un texte, produite
par un modèle de langage...

[Extrait 2 - page 4]
La dimension des vecteurs dépend du modèle : 1024 pour mistral-embed,
3072 pour text-embedding-3-large d'OpenAI...

... (etc)
```

Le prompt final envoyé à Mistral :

```
Tu es un assistant expert...
RÈGLES STRICTES : ...
CONTEXTE : [Extrait 1 - page 3] ... [Extrait 2 - page 4] ... etc.
QUESTION : Qu'est-ce qu'un embedding ?
RÉPONSE :
```

**Étape 5 — Génération**

Mistral renvoie (par exemple) :

> *Un embedding est une représentation vectorielle d'un texte : le texte
> est transformé en une liste de nombres (vecteur) par un modèle de
> langage spécialisé. Cette représentation capture le sens du texte,
> de sorte que deux textes sémantiquement proches auront des vecteurs
> proches. La dimension typique varie selon le modèle — par exemple,
> mistral-embed produit des vecteurs de dimension 1024.*

**Étape 6 — Affichage**

Streamlit affiche la réponse, suivie d'un `st.expander` avec les 4
extraits sources. L'utilisateur peut cliquer dessus pour vérifier
que la réponse correspond bien au document.

---

## 8. Pistes d'amélioration

Pour aller plus loin (utile à mentionner dans la conclusion du rapport) :

1. **Évaluation quantitative** — construire un jeu de questions/réponses
   de référence et mesurer objectivement la qualité (exact match, F1,
   taux d'hallucination).

2. **Mémoire conversationnelle** — aujourd'hui chaque question est
   traitée indépendamment. On pourrait garder le contexte des échanges
   précédents pour gérer les questions de suivi (« et celui-là ? »,
   « peux-tu préciser ? »).

3. **Reranking** — après la recherche FAISS, rerunner un modèle plus
   fin (cross-encoder) sur les top-20 pour trier plus finement. Améliore
   souvent la qualité au prix d'un peu plus de latence.

4. **OCR pour les PDF scannés** — intégrer Tesseract pour traiter
   les documents image.

5. **Extraction de tableaux** — utiliser `pdfplumber` en complément
   pour les PDF riches en tableaux.

6. **Multi-document** — indexer plusieurs PDF à la fois, avec
   possibilité de filtrer par source dans la recherche.

7. **Fine-tuning du prompt** — tester des variantes du prompt système
   et mesurer l'impact sur la qualité.

8. **Cache d'embeddings** — si un chunk apparaît dans plusieurs
   documents (ou si le même document est ré-indexé), éviter de
   re-appeler l'API Mistral.

---

## 9. Glossaire

- **API** : interface qui permet à un programme de communiquer avec un
  autre, typiquement un service distant via Internet.
- **Chunk** : petit morceau de texte issu du découpage d'un document
  plus grand.
- **Chunk_overlap** : nombre de caractères partagés entre deux chunks
  consécutifs (pour ne pas perdre d'information aux frontières).
- **Chunk_size** : taille maximale d'un chunk en caractères.
- **Embedding** : liste de nombres (vecteur) qui représente le sens
  d'un texte ; produite par un modèle d'embedding.
- **FAISS** : bibliothèque open-source de Meta pour la recherche rapide
  de vecteurs similaires (Facebook AI Similarity Search).
- **Framework** : ensemble d'outils et de briques logicielles réutilisables
  qui structure le développement d'une application.
- **Hallucination** : réponse d'un LLM qui semble correcte mais est
  factuellement fausse.
- **LangChain** : framework Python pour orchestrer les composants d'une
  application LLM (chargement de documents, embeddings, bases
  vectorielles, prompts, modèles).
- **LLM (Large Language Model)** : modèle de langage de grande taille
  capable de générer du texte ; ex : ChatGPT, Claude, Mistral.
- **Mistral** : startup française d'IA, dont les modèles de langage
  (embedding + génération) sont utilisés via API dans ce projet.
- **Prompt** : texte d'instruction donné au LLM avant qu'il produise
  sa réponse.
- **Prompt engineering** : art de rédiger des prompts efficaces pour
  obtenir le comportement voulu d'un LLM.
- **PyPDF** : bibliothèque Python pour lire et manipuler les PDF.
- **RAG (Retrieval-Augmented Generation)** : architecture combinant
  recherche documentaire et génération par LLM pour produire des
  réponses ancrées dans un document source.
- **Retriever** : composant qui cherche les chunks pertinents dans la
  base vectorielle à partir d'une question.
- **Similarité cosinus** : mesure de proximité entre deux vecteurs
  (entre -1 et 1) ; 1 = identiques en sens, 0 = sans rapport, -1 =
  opposés.
- **Streamlit** : framework Python pour créer des applications web
  interactives en quelques lignes de code.
- **Token** : unité élémentaire manipulée par un LLM (≈ un mot ou un
  morceau de mot).
- **Vector store (base vectorielle)** : structure de données spécialisée
  dans le stockage et la recherche rapide de vecteurs d'embeddings.

---

*Ce guide a été rédigé pour accompagner un projet étudiant. Il peut être
étendu avec les résultats d'évaluation une fois la Phase 6 du plan de
travail réalisée.*
