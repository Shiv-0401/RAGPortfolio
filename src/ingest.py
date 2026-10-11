"""Load documents from data/documents/, chunk them, embed them, and store in ChromaDB."""

import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "shiv_portfolio"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []

    txt_loader = DirectoryLoader(
        DOCUMENTS_DIR, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        DOCUMENTS_DIR, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader,
        show_progress=False,
    )
    documents.extend(md_loader.load())

    pdf_loader = DirectoryLoader(
        DOCUMENTS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=False,
    )
    documents.extend(pdf_loader.load())

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    vectorstore.add_documents(chunks)
    return vectorstore


def main():
    print(f"Loading documents from {DOCUMENTS_DIR} ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print("No documents found. Add .txt, .md, or .pdf files to data/documents/.")
        return

    print("Splitting documents into chunks (size=500, overlap=50)...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Generating embeddings with {EMBEDDING_MODEL} and storing in ChromaDB...")
    vectorstore = build_vectorstore(chunks)

    count = vectorstore._collection.count()
    print(f"Indexed {count} chunks into ChromaDB at {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
