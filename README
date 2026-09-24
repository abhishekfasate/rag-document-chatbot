# RAG Document Chatbot

A document-based AI chatbot that answers user questions using **Retrieval-Augmented Generation (RAG)**.

The application uses **LangChain to manage the RAG workflow, Sentence Transformers (`all-MiniLM-L6-v2`)** for document embeddings, and **ChromaDB** as the vector database.

## Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* Sentence Transformers
* `all-MiniLM-L6-v2`
* RAG (Retrieval-Augmented Generation)
* LLM

## How It Works

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
User Question
    ↓
Retriever
    ↓
Relevant Document Chunks
    ↓
LLM
    ↓
Final Answer
```

## Project Structure

```text
rag-document-chatbot/
│
├── static/
│   └── pdf/txt doc only supported
│
├── chroma_db/
│
├── constants/
│   └── ...
│
├── ...
│
├── requirements.txt
└── README.md
```

> The exact project structure may vary depending on the implementation.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abhishekfasate/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Setup

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Add  Documents:

Place the documents you want the chatbot to answer questions from inside the `static` folder.

```text
static/
├── document1.pdf
├── document2.txt
├── company_policy.pdf
└── product_information.txt
```

## Configure the Document Path

The document folder path is configured using the `FILE_PATH` variable.

```python
FILE_PATH = "static/"
```

## Run the Application

After adding your documents to the `static` folder, run the application using the appropriate Python entry point.

## RAG Pipeline

The application follows these main steps:

### 1. Document Ingestion

Documents are loaded from the configured `FILE_PATH`.

### 2. Chunking

Large documents are divided into smaller chunks so that relevant sections can be retrieved efficiently.

### 3. Embeddings

Each document chunk is converted into a vector using:

```text
all-MiniLM-L6-v2
```

### 4. Vector Storage

The generated embeddings are stored in **ChromaDB**.

### 5. Retrieval

When a user asks a question, the application searches ChromaDB for the most relevant document chunks.

### 6. Generation

The retrieved context is passed to the LLM, which generates the final answer based on the relevant documents.

**This project is for learning and demonstration purposes.**
