import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import clean_text


# --------------------------------------------------
# Load TF-IDF vectorizer
# --------------------------------------------------

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

models = {
    "Logistic Regression": joblib.load(
        "models/logistic_model.pkl"
    ),

    "SVM": joblib.load(
        "models/svm_model.pkl"
    ),

    "Random Forest": joblib.load(
        "models/random_forest_model.pkl"
    )
}


# --------------------------------------------------
# Load Politifact validation data
# --------------------------------------------------

fake_data = pd.read_csv(
    "dataset/validation/politifact_fake.csv"
)

real_data = pd.read_csv(
    "dataset/validation/politifact_real.csv"
)

fake_data["label"] = 0
real_data["label"] = 1

validation_data = pd.concat(
    [fake_data, real_data],
    ignore_index=True
)

validation_data = validation_data.dropna(
    subset=["title"]
)

validation_data = validation_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Prepare titles
# --------------------------------------------------

validation_data["clean_title"] = (
    validation_data["title"]
    .astype(str)
    .apply(clean_text)
)

X_validation = tfidf.transform(
    validation_data["clean_title"]
)

y_validation = validation_data["label"]


# --------------------------------------------------
# Test every model
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_validation,
            predictions
        )
    )

    print("\nPrediction Distribution:")

    print(
        pd.Series(
            predictions
        ).value_counts()
    )


print("\n" + "=" * 60)
print("ALL MODEL VALIDATION COMPLETED")
print("=" * 60)