import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# Load processed dataset
# =========================================================

df = pd.read_csv("dataset/processed_news.csv")

print("Processed dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# Prepare input and output
# =========================================================


X = df["content"]

y = df["label"]

print("\nClass distribution:")
print(y.value_counts())

print("\n0 = Fake News")
print("1 = Real News")


# =========================================================
# Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# TF-IDF
# =========================================================

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# Save vectorizer
joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


# =========================================================
# Evaluation Function
# =========================================================

results = []


def evaluate_model(model, model_name):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nAccuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Save confusion matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )

    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"

    plt.savefig(
        "results/" + filename,
        dpi=150
    )

    plt.close()

    return model


# =========================================================
# Logistic Regression
# =========================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model = evaluate_model(
    logistic_model,
    "Logistic Regression"
)

joblib.dump(
    logistic_model,
    "models/logistic_model.pkl"
)


# =========================================================
# Naive Bayes
# =========================================================

naive_bayes_model = MultinomialNB()

naive_bayes_model = evaluate_model(
    naive_bayes_model,
    "Naive Bayes"
)


# =========================================================
# Linear SVM
# =========================================================

svm_model = LinearSVC()

svm_model = evaluate_model(
    svm_model,
    "SVM"
)

# Save SVM model
joblib.dump(
    svm_model,
    "models/svm_model.pkl"
)


# =========================================================
# Random Forest
# =========================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model = evaluate_model(
    random_forest_model,
    "Random Forest"
)

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)


# =========================================================
# Model Comparison
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


# Save comparison CSV

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# =========================================================
# Model Comparison Graph
# =========================================================

results_plot = results_df.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xlabel("Models")

plt.ylim(0, 1.05)

plt.xticks(rotation=0)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=150
)

plt.close()


# Copy graph for website

import shutil

shutil.copy(
    "results/model_comparison.png",
    "static/model_comparison.png"
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModels trained:")
print("1. Logistic Regression")
print("2. Naive Bayes")
print("3. Linear SVM")
print("4. Random Forest")

print("\nSaved files:")
print("- models/tfidf_vectorizer.pkl")
print("- models/logistic_model.pkl")
print("- models/svm_model.pkl")
print("- models/random_forest_model.pkl")
print("- results/model_comparison.csv")
print("- results/model_comparison.png")