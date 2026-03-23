import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "./research_papers"
DB_DIR = "./vector_db_storage"

for d in [DATA_DIR, DB_DIR]:
    if not os.path.exists(d): os.makedirs(d)

@st.cache_resource(show_spinner=False)
def get_vector_db():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3", 
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        chunk_size=64 
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings), embeddings