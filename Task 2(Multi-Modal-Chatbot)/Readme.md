# 🧠 Multi-Modal Chatbot with Intent Classifier

This project is a local multi-modal chatbot that integrates text, image, and intent-based responses using:
- 🧠 **LLaMA3** for general text response
- 🖼️ **LLaVA** for image analysis
- 🎨 **Stable Diffusion** for text-to-image generation
- 🧪 **Intent Classifier (RandomForest)** for intelligent input routing

---

## 📁 Folder Structure

TASK 2/
│
├── App/
│ └── app.py ← Streamlit GUI (uses trained classifier)
│
├── Model/
│ ├── model.ipynb ← Jupyter notebook for model training
│ ├── chatbot_intent_classifier.pkl ← Trained RandomForest model
│ └── tfidf_vectorizer.pkl ← TF-IDF vectorizer
│
├── requirements.txt ← All dependencies
└── README.md ← Project overview (this file)


---

## ⚙️ Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
🚀 How to Run the App
From inside the App/ folder:

streamlit run app.py
Make sure:

LLaVA and LLaMA are running locally through Ollama

Stable Diffusion WebUI is active at http://127.0.0.1:7860

📊 Model Details
The intent classifier was trained on 90 prompts across 3 classes:

image_description

image_generation

general_question

It uses:

TF-IDF Vectorizer (1-2 grams)

Random Forest Classifier (200 estimators)

Achieved ✅ >90% accuracy on test set

📌 Saved models are located in the Model/ folder.

🧠 How Intent Classifier Helps
The classifier enables:

Smarter routing of user prompts to correct modules

Generalization beyond simple keyword matching

Better understanding of user inputs like "make me a mountain scene" or "explain this photo"

🗂️ Notes for Evaluation
📁 Models are saved and included

📈 Accuracy, confusion matrix, and metrics shown in model.ipynb

✅ Trained model is under 100MB and included in the repo

💬 GUI is built using Streamlit (app.py)
🧩 This project is built as an extension of the chatbot base covered earlier in the course, now enhanced with multi-modal input/output and an intent classifier for routing user prompts.