import pandas as pd
import pickle
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load the pre-trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer from disk."""
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Step 2: Preprocess user input
def preprocess_input(text):
    """Clean and preprocess the input text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Step 3: Predict sentiment
def predict_sentiment(model, vectorizer, review):
    """Predict whether the sentiment is positive or negative."""
    review = preprocess_input(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😢"

# Step 4: Create the Streamlit interface
def run_sentiment_app():
    st.set_page_config(page_title="Sentiment Analysis Tool", layout="centered")
    
    # Load the trained model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Set up Streamlit title and instructions
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("""
    **Analyze the sentiment of a movie review.**  
    Type a review, and the model will classify it as positive or negative.
    """)
    
    # Placeholder example
    review = st.text_input(
        "Enter a movie review:",
        placeholder="Write a review like: 'This movie was fantastic, I loved it!'"
    )
    
    # Button to analyze sentiment
    if st.button('Analyze Sentiment'):
        if review.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            with st.spinner("Analyzing... Please wait."):
                sentiment = predict_sentiment(model, vectorizer, review)
                st.subheader(f"Sentiment: {sentiment}")
                st.success(f"The review is classified as: **{sentiment}**")

    # Footer with extra info
    st.markdown("---")
    st.write("🔍 **Tip**: Try writing reviews like 'The acting was terrible' or 'Amazing storyline!'")

# Main function to run the Streamlit app
if __name__ == "__main__":
    run_sentiment_app()