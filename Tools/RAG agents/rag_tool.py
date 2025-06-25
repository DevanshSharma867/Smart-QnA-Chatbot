import os
from dotenv import load_dotenv
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
VECTORDB_DIR = "data/csv_vectordb"
K = 3

def lookup_rag_information(query: str) -> str:
    """Search within the VECTORDB to find the relevant information. Input should be a search query."""
    vectordb = Chroma(
        collection_name="csv-rag-chroma",
        persist_directory=str(here(VECTORDB_DIR)),
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )
    docs = vectordb.similarity_search(query, k=K)
    return "\n\n".join([doc.page_content for doc in docs]) 