"""Build a LangChain RAG chain combining the ChromaDB retriever and the Groq LLM."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import get_retriever

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = """You are an AI assistant helping recruiters and interviewers learn about Shiv Patel, a CS and Data Science student at Rutgers University. Answer questions based on the provided context about his projects, skills, and experience. Be specific and enthusiastic about his work. If the answer is not in the context, say so honestly.

Context: {context}

Question: {question}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    retriever = get_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask(question):
    """Run the RAG chain and return both the answer and the source documents used."""
    rag_chain, retriever = build_chain()

    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)

    sources = []
    seen = set()
    for doc in source_docs:
        name = os.path.basename(doc.metadata.get("source", "unknown"))
        if name not in seen:
            seen.add(name)
            sources.append(name)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What projects has Shiv built?"
    result = ask(question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
