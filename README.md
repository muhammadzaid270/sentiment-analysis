# Sentiment Analysis Model

This Sentiment Analysis model is built to classify movie reviews as either **positive** or **negative**. The model uses a **Logistic Regression** classifier and **TF-IDF** for feature extraction, trained on the IMDB dataset. The input review text is preprocessed (removal of non-alphabetic characters and conversion to lowercase) and transformed using the TF-IDF vectorizer to make predictions.

## How to Access the Web App

To interact with the model, you can visit the web app by clicking on the following link:  
[sentiment-analysis-01.streamlit.app](http://sentiment-analysis-01.streamlit.app)

In the web app, you can enter a movie review into the text box and click "Analyze" or Press Enter. The model will return whether the sentiment of the review is **positive** or **negative** based on its content.

## How to Run the Model Locally

1. **Clone the Repository**:
   To use the model on your local machine, clone the repository using the following command:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git

2. **Install Dependencies**: Navigate to the project folder and install the required libraries:
   ```bash
   cd sentiment-analysis
   pip install -r requirements.txt

3. **Prepare the Dataset**: Download the IMDB dataset (`imdb.csv`) and place it in the project directory.

4. **Run the Model**: Once everything is set up, you can run the script:
   ```bash
   python sentiment_analysis.py

This will launch a **Tkinter** GUI that allows you to input movie reviews and analyze their sentiment.