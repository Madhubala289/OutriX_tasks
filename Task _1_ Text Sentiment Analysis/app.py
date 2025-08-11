import streamlit as st
import joblib

# Load the trained model
model = joblib.load("models/best_model.joblib")

st.set_page_config(page_title="Movie Review Sentiment Classifier", layout="centered")
st.title("🎬 Movie Review Sentiment Classifier")
st.write("This model predicts whether a movie review is **Positive** or **Negative**.")

review_text = st.text_area("Enter a movie review here:")

if st.button("Classify"):
    if review_text.strip():
        prediction = model.predict([review_text])[0]
        probability = model.predict_proba([review_text])[0]
        classes = model.classes_
        st.subheader(f"Prediction: {prediction.capitalize()}")
        st.write("Confidence:")
        for cls, prob in zip(classes, probability):
            st.write(f"- {cls.capitalize()}: {prob:.2f}")
    else:
        st.warning("Please enter some text to classify.")
