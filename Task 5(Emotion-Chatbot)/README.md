
# 🤖 Emotion-Aware Chatbot

## 📌 Overview

This project integrates **sentiment analysis** into a chatbot to detect and respond appropriately to customer **emotions** (joy, sadness, anger, fear, love, and surprise) during interactions. The chatbot uses a fine-tuned BERT model for emotion classification and dynamically responds based on the detected **sentiment** — *positive*, *negative*, or *neutral*.

---

## 🎯 Task Objective

* **Goal:** Integrate sentiment/emotion detection into a chatbot for improved emotional intelligence.
* **Expected Outcome:**

  * Detect user emotion with high accuracy.
  * Respond accordingly to maintain high customer satisfaction.
* **Achieved Metrics:**

  * ✅ **Accuracy**: 92.35%
  * ✅ **Precision**: 92.32%
  * ✅ **Recall**: 92.35%
  * ✅ **F1 Score**: 92.31%

---

## 🗂 Project Structure

```
Task_5/
├── App/
│   ├── main.py               # Streamlit chatbot interface
│   └── emotion_bot.py        # Emotion detection + response logic
│
├── Model/
│   ├── model.ipynb           # Model training code (Jupyter Notebook)
│   └── training.ipynb        # Additional training workflow (if any)
│
├── my_emotion_bot/           # Saved model and tokenizer
│   ├── config.json
│   ├── model.safetensors     # 🔗 See below for download
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── requirements.txt          # Required libraries
├── test.py                   # Model testing script
└── README.md                 # Project documentation
```

---

## 🧠 Model Details

* **Architecture**: BERT (Pretrained: `bert-base-uncased`)
* **Fine-tuned for**: 6 emotion classes (sadness, joy, love, anger, fear, surprise)
* **Evaluation**:

  * 📉 Confusion Matrix shows well-distributed performance across all classes.

![Confusion Matrix](my_emotion_bot/confusion_matrix.png) <!-- Or upload to GitHub and use a public link -->

---

## 🧪 Demo

A simple chatbot interface built using **Streamlit** that:

* Accepts user messages.
* Detects emotion → maps to sentiment.
* Generates sentiment-aware responses.
* Displays emotion, sentiment, and confidence.

---

## 💾 Download Model Weights

📦 **Model Weights (model.safetensors)**:
[Download from Google Drive]([https://drive.google.com/your-model-link](https://drive.google.com/file/d/1_EGv-lEm6k95F-en6sfB0BFRZlYmaDBo/view?usp=sharing) <!-- Replace with your actual model upload link -->

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-chatbot.git
cd emotion-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the chatbot

```bash
streamlit run App/main.py
```

---

## ✅ Requirements

See [`requirements.txt`](requirements.txt)

<details>
<summary>Click to view contents</summary>

```txt
transformers
torch
streamlit
numpy
```

</details>

---

## 🧪 Evaluation Criteria

* ✔️ Minimum Accuracy (70%) – **Achieved: 92.35%**
* ✔️ Precision, Recall, F1-score showcased
* ✔️ Confusion Matrix provided
* ✔️ Sentiment-aware chatbot responses
* ✔️ GUI included (via Streamlit)
* ✔️ Proper folder structure & documentation
* ✔️ Model + tokenizer saved
* ✔️ All files integrated with course framework

---

## 🔗 Submission Checklist

* ✅ `requirements.txt` included
* ✅ Model trained using `.ipynb`
* ✅ Model weights saved (`.safetensors`)
* ✅ Streamlit GUI implemented
* ✅ GitHub repo ready
* ✅ Emotion model integrated with chatbot

---

## 🔄 Integration Note

This chatbot is not a standalone project. It has been integrated within the framework of the course curriculum and builds upon previous modules on chatbot interfaces and NLP pipelines.

