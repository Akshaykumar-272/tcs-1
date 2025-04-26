import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Load models once ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

# --- Load CSV and vector index ---
@st.cache_data
def load_csv_and_index():
    df = pd.read_csv(r"C:\Users\Yogi\Desktop\TCS\data_synthetic.csv")
    text_data = [" | ".join([f"{col}: {str(row[col])}" for col in df.columns]) for _, row in df.iterrows()]
    vectors = embedder.encode(text_data)
    index = faiss.IndexFlatL2(384)
    index.add(np.array(vectors))
    return df, text_data, index, vectors

df, text_data, index, vectors = load_csv_and_index()

# --- UI ---
st.set_page_config(page_title="CSV Insurance Bot")
st.title("🤖 CSV-Based Insurance Assistant")

query = st.text_input("Ask a question about customer or policy data:")

if query:
    with st.spinner("Thinking..."):
        query_vec = embedder.encode([query])
        D, I = index.search(np.array(query_vec), k=3)
        context = " ".join([text_data[i] for i in I[0]])

        answer = qa_pipeline(question=query, context=context)

        if answer["score"] < 0.2 or answer["answer"].strip() == "":
            st.warning("🤔 I couldn't find a confident answer. Try rephrasing your question.")
        else:
            st.success(f"🧠 Answer: {answer['answer']}")
