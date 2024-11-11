import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv('imdb.csv')
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
    # model.fit(X_train, y_train)

    # with open('sentiment_model.pkl', 'wb') as model_file:
    #     pickle.dump(model, model_file)
    # with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    #     pickle.dump(vectorizer, vec_file)

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

# Step 5: Create a GUI using tkinter with enhanced resolution and style
def run_sentiment_gui():
    # Load the trained model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Function to handle button click
    def analyze_sentiment():
        review = entry.get()
        if review.strip() == "":
            messagebox.showwarning("Input Error", "Please enter a review.")
        else:
            sentiment = predict_sentiment(model, vectorizer, review)
            result_label.config(
                text=f"Sentiment: {sentiment}",
                fg="#4CAF50" if sentiment == "positive" else "#F44336"
            )

    # Set up the tkinter window
    window = tk.Tk()
    window.title("Sentiment Analysis")
    window.geometry("600x400")  # Increased resolution
    window.configure(bg="#f0f0f5")  # Light grey background

    # Add a title label
    title_label = tk.Label(
        window,
        text="Sentiment Analysis Tool",
        font=("Helvetica", 18, "bold"),
        fg="#333333",
        bg="#f0f0f5"
    )
    title_label.pack(pady=20)

    # Add a prompt label
    prompt_label = tk.Label(
        window,
        text="Enter a movie review to analyze:",
        font=("Arial", 14),
        bg="#f0f0f5"
    )
    prompt_label.pack(pady=10)

    # Add an entry box
    entry = tk.Entry(window, width=50, font=("Arial", 12), borderwidth=2, relief="groove")
    entry.pack(pady=10)

    # Add an "Analyze" button
    analyze_button = tk.Button(
        window,
        text="Analyze",
        command=analyze_sentiment,
        font=("Arial", 14, "bold"),
        bg="#007BFF",
        fg="white",
        activebackground="#0056b3",
        activeforeground="white",
        cursor="hand2",
        borderwidth=0,
        relief="flat"
    )
    analyze_button.pack(pady=20)

    # Add a result label
    result_label = tk.Label(
        window,
        text="",
        font=("Helvetica", 16),
        bg="#f0f0f5"
    )
    result_label.pack(pady=20)

    # Run the window loop
    window.mainloop()

# Main function to run the program
if __name__ == "__main__":
    # Provide the correct path to the dataset
    dataset_path = 'C:/path_to_your_dataset/imdb.csv'  # Update this path
    X, y = load_and_preprocess_data(dataset_path)
    train_and_save_model(X, y)  # This will not print anything in the terminal

    # Run the GUI application
    run_sentiment_gui()