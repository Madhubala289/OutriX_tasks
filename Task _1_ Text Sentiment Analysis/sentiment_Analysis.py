# sentiment_Analysis.py
import nltk
import re
import string
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy

from nltk.corpus import movie_reviews, stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

STOPWORDS = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.text not in STOPWORDS and not token.is_space]
    return " ".join(lemmas)

# Prepare dataset from NLTK movie_reviews
docs = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append(movie_reviews.raw(fileid))
        labels.append(category)

df = pd.DataFrame({"text": docs, "label": labels})
df["clean_text"] = df["text"].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

def train_and_evaluate(model, model_name):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)),
        ("clf", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n==== {model_name} ====")
    print(classification_report(y_test, preds))

    # Save metrics
    report = classification_report(y_test, preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"results/{model_name}_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/{model_name}_confusion_matrix.png")
    plt.close()

    return pipe, acc

# Train models
nb_model, nb_acc = train_and_evaluate(MultinomialNB(), "NaiveBayes")
lr_model, lr_acc = train_and_evaluate(LogisticRegression(max_iter=500), "LogisticRegression")

# Save best model
best_model = lr_model if lr_acc >= nb_acc else nb_model
joblib.dump(best_model, "models/best_model.joblib")
print(f"\nBest model saved: {'LogisticRegression' if lr_acc >= nb_acc else 'NaiveBayes'}")
