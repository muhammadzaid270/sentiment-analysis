import pandas as pd
import pickle
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabetic characters
    df['review'] = df['review'].str.lower()  # Convert to lowercase
    X = df['review']
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Convert labels to 0 (negative) and 1 (positive)
    return X, y

# Step 2: Train the model and save it (this will not print anything)
def train_and_save_model(X, y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_transformed = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open('sentiment_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

# Step 3: Load the model and vectorizer for prediction
def load_model_and_vectorizer():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Step 4: Preprocess user input and make predictions
def preprocess_input(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def predict_sentiment(model, vectorizer, review):
    review = preprocess_input(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    return "positive" if prediction == 1 else "negative"

# Step 5: Create the Streamlit interface
def run_sentiment_app():
    # Load the trained model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Set up Streamlit title and instructions
    st.title("Sentiment Analysis Tool")
    st.write("Enter a movie review to analyze the sentiment.")

    # Get user input (text review)
    review = st.text_input("Movie Review:")

    # Button to analyze sentiment
    if st.button('Analyze'):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment = predict_sentiment(model, vectorizer, review)
            st.subheader(f"Sentiment: {sentiment}")
            st.write(f"The review is classified as: **{sentiment}**")

# Main function to run the Streamlit app
if __name__ == "__main__":
    dataset_path = 'imdb.csv'  # Ensure you upload this dataset along with the app
    X, y = load_and_preprocess_data(dataset_path)
    train_and_save_model(X, y)  # Run this once to generate the model files

    # Run the Streamlit app
    run_sentiment_app()