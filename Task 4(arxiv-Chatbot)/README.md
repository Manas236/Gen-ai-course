
# 🤖 arXiv CS Expert Chatbot

This project is an AI-powered chatbot that helps users explore and understand scientific papers from the arXiv Computer Science (CS) collection. It supports semantic search and provides intelligent explanations and summaries of top-matching papers. The backend uses sentence embeddings and a local LLM served via Ollama for query understanding and response generation.

---

## 📁 Project Structure

```

TASK 4/
├── App/
│   └── app.py                 # Streamlit interface
├── model/
│   └── arxiv\_embeddings.pkl   # Precomputed sentence embeddings
├── notebooks/
│   ├── model\_training.ipynb   # Generates embeddings using SPECTER model
│   ├── model\_accuracy.ipynb   # Evaluates classifier accuracy
├── arxiv-metadata-oai-snapshot.json  # Full metadata of arXiv papers
├── requirements.txt           # Dependencies
└── README.md                  # This file

````

---

## ⚙️ Features

- 🔍 **Semantic Paper Search**: Uses `sentence-transformers` to find top relevant papers for any query.
- 🧠 **LLM Explanation & Summary**: Summarizes top paper and explains based on user question via Ollama.
- 📈 **Model Accuracy**: Evaluated with Logistic Regression on paper category prediction.
- 💬 **Interactive Chat Interface**: Built with Streamlit for ease of use.

---

## 🧪 Evaluation Results

Model was trained to classify papers into 8 grouped CS domains:
- `AI`, `NLP`, `ML`, `CV`, `DS`, `SE`, `Security`, `Networking`

| Metric        | Value        |
|---------------|--------------|
| ✅ Accuracy   | **86.06%**   |
| 🎯 Precision  | **0.85**     |
| 📥 Recall     | **0.83**     |

Confusion matrix and full classification report are visualized inside `model_accuracy.ipynb`.

---

## 🔧 Setup Instructions

1. **Install Dependencies**

```bash
pip install -r requirements.txt
````

2. **Run the Chatbot Interface**

```bash
streamlit run App/app.py
```

3. **Ensure Ollama is Running Locally**
Ollama must be preinstalled and running (https://ollama.com/).

Make sure `llama3` model is active in Ollama:

```bash
ollama run llama3
```

---

## 📂 Embedding Pipeline

1. The `model_training.ipynb` loads and filters CS papers from `arxiv-metadata-oai-snapshot.json`.
2. It uses the `"allenai-specter"` sentence transformer to generate embeddings.
3. Data is saved to `model/arxiv_embeddings.pkl` for use by the chatbot.

---

## 🧠 Classification Model

`model_accuracy.ipynb` evaluates how well embeddings can be used for category prediction:

* Uses `LogisticRegression`
* Achieves **86.06% accuracy**
* Evaluated using precision, recall, and confusion matrix

---

## 📌 Requirements

* `streamlit`
* `sentence-transformers`
* `scikit-learn`
* `torch`
* `pandas`
* `numpy`
* `matplotlib`
* `transformers`

---

## 📣 Credits

* **Dataset**: [arXiv Metadata Snapshot](https://www.kaggle.com/Cornell-University/arxiv)
* **Model**: [`allenai-specter`](https://huggingface.co/allenai/specter) for semantic embedding
* **LLM**: [`llama3`](https://ollama.com/library/llama3) via Ollama (local)

---

## ✅ Status

> 🚀 **Project Complete**
> Achieves over 70% accuracy with strong classification performance and interactive capabilities.
