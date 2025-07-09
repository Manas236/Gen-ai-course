


# GEN---AI-COURSE

A curated set of AI-powered projects showcasing applications of NLP, multi-modal learning, document retrieval, and semantic search.

---

## 📁 Project Structure

```

GEN---AI-COURSE/
├── customer_service_chatbot_LLM/
│ ├── dataset/
│ ├── src/
│ ├── README.md
│ └── requirements.txt
│
├── LLM/
│ ├── End-To-End-Gemini-Project-main/
│ ├── llama3/
│ ├── intro_palm_api.ipynb
│ └── python.ipynb
│
├── NLP/
│ ├── NLP_bot/
│ ├── NLP (1).ipynb
│ ├── NLP_Word2vec.ipynb
│ └── SMSSpamCollection.txt
│
├── Task 1 (Article-Generator)/
│   ├── app/
│   ├── README.md
│   └── requirements.txt
│
├── Task 2 (Multi-Modal-Chatbot)/
│   ├── App/
│   │   ├── .env
│   │   └── app.py
│   ├── Model/
│   │   ├── chatbot\_intent\_classifier.pkl
│   │   ├── model.ipynb
│   │   └── tfidf\_vectorizer.pkl
│   ├── Visuals/
│   ├── Readme.md
│   └── requirements.txt
│
├── Task 3 (MedQuAD-Chatbot)/
│   ├── App/
│   │   ├── app.py
│   │   └── label\_encoder.pkl
│   │
│   ├── MedQuAD/
│   ├── Model/
│   │   ├── label\_encoder.pkl
│   │   ├── medquad.json
│   │   ├── rf\_tfidf\_model.pkl
│   │   ├── tfidf\_vectorizer.pkl
│   │   └── train\_medquad\_model.ipynb
│   ├── confusion\_matrix.png
│   ├── new\_parse.py
│   ├── README.md
│   └── requirements.txt
│
├── Task 4 (arxiv-Chatbot)/
│   ├── App/
│   │   └── app.py
│   ├── model/
│   │   ├── arxiv\_embeddings\_drive\_link.txt
│   │   └── arxiv\_embeddings.pkl
│   ├── notebooks/
│   │   ├── model\_accuracy.ipynb
│   │   └── model\_training.ipynb
│   ├── resources/
│   │   ├── arxiv-metadata-oai-snapshot\_drive\_link.txt
│   │   └── arxiv-metadata-oai-snapshot.json
│   ├── visuals/
│   │   └── Confusion\_matrix.png
│   ├── README.md
│   └── requirements.txt
├── Task 5 (Emotion-Chatbot)/
│ ├── App/
│ ├── .env
│ ├── main.py
│ └── Model/
│ ├── emotion_dataset/
│ ├── model.ipynb
│ └── training.ipynb


````

---

## 📌 Project Tasks

### 🔹 Task 1: Article Generator
- Generates AI-written articles based on user-provided topics or prompts.
- NLP techniques like keyword extraction and summarization used.
- Streamlit-based frontend for user interaction (`app/` directory).
- Modular design for easy integration of new generation models.

### 🔹 Task 2: Multi-Modal Chatbot
- Integrates both text and potential vision-based input (if extended).
- Uses a trained intent classification model (`chatbot_intent_classifier.pkl`) with TF-IDF vectorization.
- Frontend built with Streamlit (`app.py`).
- Dependencies listed in `requirements.txt`.

### 🔹 Task 3: MedQuAD Chatbot
- Q&A bot trained on NIH MedQuAD dataset.
- Uses TF-IDF with Random Forest classification.
- Includes semantic label encoder and JSON-based question-answer dataset.
- Accuracy and confusion matrix visual included.

### 🔹 Task 4: arxiv Chatbot
- Designed for document summarization and semantic search on arXiv papers.
- Embeddings generated and stored (`arxiv_embeddings.pkl`).
- Metadata handled via `arxiv-metadata-oai-snapshot.json`.
- Model training and evaluation in dedicated notebooks.

### 🔹 Task 5: Emotion-Based Chatbot
- Detects emotions in user input (e.g., joy, anger, sadness) using a trained classifier.
- Provides emotionally aware responses to enhance user experience.
- Includes `Streamlit-based interaction (`main.py`).
- Dataset and training notebooks available under the `Model/` directory.

---

## 💻 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/GEN---AI-COURSE.git
cd GEN---AI-COURSE

# Install requirements for a specific task
cd "Task 3 (MedQuAD-Chatbot)"
pip install -r requirements.txt

# Run the Streamlit app
streamlit run App/app.py
````

---

## 📜 License

This project is provided for academic and learning purposes.

---

## 🙏 Acknowledgements

* NIH MedQuAD Dataset
* arXiv Open Access Metadata
* HuggingFace Transformers
* Scikit-learn
* Streamlit

```

Let me know if you'd like me to **create individual `README.md` files** for each task folder too.
```
