import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

with open("../Model/arxiv_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    ids = data["ids"]
    texts = data["texts"]
    embeddings = data["embeddings"]

model = SentenceTransformer("allenai-specter")

st.set_page_config(page_title="arXiv CS Chatbot", layout="wide")
st.title("🤖 arXiv CS Expert Chatbot")
st.markdown("Ask complex questions about computer science research papers from arXiv.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def search_papers(query, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(ids[i], texts[i], similarities[i]) for i in top_indices]
    return results

def query_llm_ollama(prompt, model="llama3"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
    except Exception as e:
        return f"[LLM Error(Please try again this is not the erro you should but if you are getting this after multiple attempts. You might not be calling your ollama model properly)] {str(e)}"

def build_prompt(question, paper_summary, history):
    context = f"Paper Summary:\n{paper_summary.strip()}\n\n"
    history_text = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]])
    return f"{context}{history_text}\nUser: {question}\nBot:"

user_input = st.text_input("💬 Ask a question about computer science:")

if user_input:
    with st.spinner("🔍 Searching relevant papers..."):
        top_papers = search_papers(user_input)

    st.subheader("📄 Top Matching Papers")
    for i, (pid, text, score) in enumerate(top_papers):
        st.markdown(f"**{i+1}. ID: {pid}** (Score: {score:.2f})")
        st.markdown(f"`{text[:500]}...`")  

    top_paper_text = top_papers[0][1]

    with st.spinner("📚 Generating summary..."):
        summary_prompt = f"Summarize the following scientific paper:\n\n{top_paper_text}"
        summary = query_llm_ollama(summary_prompt)

    st.subheader("📝 Summary")
    st.write(summary)

    with st.spinner("🤔 Generating explanation..."):
        full_prompt = build_prompt(user_input, summary, st.session_state.chat_history)
        explanation = query_llm_ollama(full_prompt)

    st.subheader("📖 Explanation")
    st.write(explanation)

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": explanation
    })

if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for turn in st.session_state.chat_history:
        st.markdown(
            f"<span style='color:#1f77b4; font-weight:bold;'>You(🙍‍♂️):</span> {turn['user']}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<span style='color:#2ca02c; font-weight:bold;'>Bot(🤖):</span> {turn['bot']}",
            unsafe_allow_html=True
        )

