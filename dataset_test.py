import pandas as pd
import joblib

from preprocess import clean_text


# Load dataset
data = pd.read_csv("dataset/news.csv")

# Load trained model and TF-IDF
model = joblib.load("models/random_forest_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def test_article(article, actual_label):

    cleaned_text = clean_text(article)
    text_tfidf = tfidf.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]

    predicted_label = "REAL" if prediction == 1 else "FAKE"
    actual = "REAL" if actual_label == 1 else "FAKE"

    confidence = max(probability) * 100

    print("\n-----------------------------")
    print("Actual:", actual)
    print("Predicted:", predicted_label)
    print("Confidence:", round(confidence, 2), "%")
    print("-----------------------------")


# Take one real article
real_article = data[data["label"] == 1].iloc[0]

# Take one fake article
fake_article = data[data["label"] == 0].iloc[0]


print("TESTING REAL ARTICLE")
test_article(
    real_article["title"] + " " + real_article["text"],
    real_article["label"]
)


print("\nTESTING FAKE ARTICLE")
test_article(
    fake_article["title"] + " " + fake_article["text"],
    fake_article["label"]
)