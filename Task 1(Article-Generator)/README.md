# 📝 Article Generator Chatbot – Model Comparison

## 📌 Task Overview

This project evaluates 3 open-source LLMs running locally via Ollama to generate a 200-word article on clean energy. The models are assessed based on length, readability, grammar, and keyword relevance.

## 🧠 Models Tested
- `tinyllama`
- `gemma:2b`
- `mistral`
## Model Weights Note

This project uses pre-trained open-source models via Ollama. No custom training was done. Models are downloaded locally using:

ollama pull tinyllama
ollama pull gemma:2b
ollama pull mistral

## 📊 Evaluation Metrics
- **Length Score**: Closeness to 200 words
- **Readability**: Flesch Reading Ease score
- **Keyword Score**: Hits for predefined keywords
- **Grammar Score**: Based on grammar mistakes
- **Total Score**: Composite average

## 🏆 Output
Final results are printed inside the notebook:
Scores for tinyllama:
  length_score: 0%
  readability_score: 18%
  keyword_score: 80%
  grammar_score: 100%
  total_score: 50%

Scores for gemma:2b:
  length_score: 98%
  readability_score: 15%
  keyword_score: 80%
  grammar_score: 100%
  total_score: 73%

Scores for mistral:
  length_score: 66%
  readability_score: 17%
  keyword_score: 80%
  grammar_score: 100%
  total_score: 66%

🏆 Recommended model for article generation: **gemma:2b**