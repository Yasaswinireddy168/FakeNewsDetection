import joblib
from preprocess import clean_text

# Load model and TF-IDF
model = joblib.load("models/random_forest_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Test news
news = """
The President announced today that the government has approved
a new national education program to provide free digital learning
resources to students across the country.
"""

# Preprocess
cleaned_news = clean_text(news)

# TF-IDF
news_tfidf = tfidf.transform([cleaned_news])

# Prediction
prediction = model.predict(news_tfidf)[0]
probabilities = model.predict_proba(news_tfidf)[0]

if prediction == 0:
    result = "FAKE NEWS"
else:
    result = "REAL NEWS"

confidence = max(probabilities) * 100

print("Prediction:", result)
print("Confidence:", round(confidence, 2), "%")