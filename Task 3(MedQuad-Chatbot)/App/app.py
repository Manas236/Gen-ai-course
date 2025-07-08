import streamlit as st

st.set_page_config(page_title="🩺 MedQuAD Chatbot", page_icon="💬")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


@st.cache_resource(show_spinner="Loading data & models …")
def load_resources():
    base_dir = Path(__file__).parent

    with open("../Model/medquad.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    df = pd.DataFrame(qa_data) 

    vectorizer = joblib.load("../Model/tfidf_vectorizer.pkl")
    clf_model = joblib.load("../Model/rf_tfidf_model.pkl")

    question_vecs = vectorizer.transform(df["question"])

    return df, vectorizer, clf_model, question_vecs

df, vectorizer, clf_model, Q_VECS = load_resources()

st.title("🩺 Medical Q&A Chatbot (MedQuAD)")
st.markdown(
    """
Type a **medical question** below.  
The app will retrieve the most relevant answer from the NIH MedQuAD dataset.  
*Note: This is for educational purposes only — not a substitute for professional advice.*
"""
)

user_q = st.text_input("Your question:", placeholder="e.g., What are the symptoms of diabetes?")
top_k = st.slider("Number of answers to display", 1, 5, value=3)

if st.button("Get answer") and user_q.strip():
    q_vec = vectorizer.transform([user_q])

    sims = cosine_similarity(q_vec, Q_VECS).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    scores = sims[idxs]

    pred_label = clf_model.predict(q_vec)[0]

    label_txt = f"Predicted Class: `{pred_label}`"  # Can map label → text if needed
    st.write(label_txt)

    if scores[0] < 0.9:
        st.warning("⚠️ Sorry, no relevant answers found.")
    else:
        for rank, (i, sim) in enumerate(zip(idxs, scores), 1):
            st.markdown(f"### 🔹 Answer #{rank}")
            st.write(f"**Matched question:** {df.loc[i, 'question']}")
            st.write("**Answer:**")
            st.info(df.loc[i, "answer"])
            st.write(f"**Semantic Group:** {df.loc[i, 'semantic_group']}")
            st.write(f"**Semantic Subgroup:** {df.loc[i, 'semantic_subgroup']}")
            st.caption(f"Similarity score: {sim:.3f}")

    st.divider()
    st.caption("Powered by MedQuAD • Model: Random Forest + TF-IDF • © 2025")

else:
    st.caption("Ask a question to see answers …")
