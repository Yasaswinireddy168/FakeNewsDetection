import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load unseen Politifact data
fake = pd.read_csv(
    "dataset/validation/politifact_fake.csv"
)

real = pd.read_csv(
    "dataset/validation/politifact_real.csv"
)

fake["label"] = 0
real["label"] = 1

data = pd.concat(
    [fake, real],
    ignore_index=True
)

# Use titles for validation
X_text = data["title"].fillna("")
y = data["label"]

# Load trained TF-IDF and SVM
tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/svm_model.pkl"
)

# Transform unseen news
X = tfidf.transform(X_text)

# Predict
predictions = model.predict(X)

# Results
accuracy = accuracy_score(
    y,
    predictions
)

print("=" * 60)
print("UNSEEN NEWS VALIDATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y,
        predictions
    )
)