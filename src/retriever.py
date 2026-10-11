"""Retrieve the most relevant chunks from ChromaDB for a given query."""

import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "shiv_portfolio"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

_embeddings = None
_vectorstore = None


def get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def get_retriever(k=TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve(query, k=TOP_K):
    """Return top-k relevant chunks with source document name and relevance score.

    Chroma similarity_search_with_relevance_scores returns a score in [0, 1]
    where higher means more relevant.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    chunks = []
    for doc, score in results:
        source_path = doc.metadata.get("source", "unknown")
        source_name = os.path.basename(source_path)
        chunks.append(
            {
                "content": doc.page_content,
                "source": source_name,
                "score": round(float(score), 4),
            }
        )
    return chunks


def indexed_document_count():
    return get_vectorstore()._collection.count()


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What projects has Shiv built?"
    for chunk in retrieve(query):
        print(f"[{chunk['source']}] score={chunk['score']}")
        print(chunk["content"][:200])
        print("---")
