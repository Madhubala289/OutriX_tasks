# 🎬Sentiment Analysis for movie review

This project classifies **movie reviews** as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning.  
It uses the **NLTK Movie Reviews dataset** and compares two models:  
- **Naive Bayes**
- **Logistic Regression**

The better-performing model is saved and used in a **Streamlit web app** for live predictions.

---

## 🚀 Features
- Preprocessing with:
  - Lowercasing
  - Removing punctuation, URLs, mentions
  - Lemmatization using spaCy
  - Stopwords removal (except negations like *not*, *never*)
- Feature extraction using **TF-IDF**
- Model training and evaluation for **Naive Bayes** and **Logistic Regression**
- Automatic saving of the **best model**
- Interactive **Streamlit** web app for predictions
- Evaluation metrics and confusion matrix plots saved in `results/`

---

## 📂 Project Structure
├── app.py # Streamlit web app
├── sentiment_Analysis.py # Training and evaluation script
├── models/ # Saved trained model
├── results/ # Metrics and plots
├── requirements.txt
└── README.md
