from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import math
from datetime import datetime

from preprocess import clean_text

app = Flask(__name__)

# =========================================================
# LOAD FINAL SVM MODEL
# =========================================================

model = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

HISTORY_FILE = "results/prediction_history.csv"


# =========================================================
# LIVE ANALYTICS
# =========================================================

def get_live_analytics():

    if not os.path.exists(HISTORY_FILE):
        return {
            "total_predictions": 0,
            "fake_count": 0,
            "real_count": 0,
            "average_confidence": 0,
            "history": []
        }

    try:
        df = pd.read_csv(HISTORY_FILE)

    except Exception:
        # If an old/corrupted history file exists,
        # start analytics fresh without affecting the model.
        return {
            "total_predictions": 0,
            "fake_count": 0,
            "real_count": 0,
            "average_confidence": 0,
            "history": []
        }

    if df.empty:
        return {
            "total_predictions": 0,
            "fake_count": 0,
            "real_count": 0,
            "average_confidence": 0,
            "history": []
        }

    # Make sure required columns exist
    required_columns = [
        "time",
        "article",
        "prediction",
        "fake_probability",
        "real_probability",
        "confidence"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    # Convert confidence to numeric
    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    fake_count = int(
        (df["prediction"] == "FAKE NEWS").sum()
    )

    real_count = int(
        (df["prediction"] == "REAL NEWS").sum()
    )

    average_confidence = df["confidence"].dropna().mean()

    if math.isnan(average_confidence):
        average_confidence = 0

    # Show latest predictions first
    history = (
        df.tail(10)
        .iloc[::-1]
        .to_dict("records")
    )

    return {
        "total_predictions": len(df),
        "fake_count": fake_count,
        "real_count": real_count,
        "average_confidence": round(
            average_confidence, 2
        ),
        "history": history
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    analytics = get_live_analytics()

    return render_template(
        "index.html",
        analytics=analytics
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Names match your existing index.html
    news_title = request.form.get(
        "news_title",
        ""
    ).strip()

    news_text = request.form.get(
        "news_text",
        ""
    ).strip()

    # Combine title + article
    combined_text = news_title + " " + news_text

    # Check empty input
    if not combined_text.strip():

        return render_template(
            "index.html",
            result=None,
            confidence=0,
            fake_probability=0,
            real_probability=0,
            analytics=get_live_analytics(),
            news_title=news_title,
            news_text=news_text
        )

    # =====================================================
    # PREPROCESS TEXT
    # =====================================================

    cleaned_text = clean_text(combined_text)

    # =====================================================
    # TF-IDF
    # =====================================================

    transformed_text = tfidf.transform(
        [cleaned_text]
    )

    # =====================================================
    # SVM PREDICTION
    # =====================================================

    prediction = model.predict(
        transformed_text
    )[0]

    # SVM decision score
    decision_score = model.decision_function(
        transformed_text
    )[0]

    # =====================================================
    # CONVERT SVM SCORE TO PROBABILITY-LIKE VALUE
    # =====================================================

    # Sigmoid transformation
    probability_real = (
        1 / (1 + math.exp(-float(decision_score)))
    )

    probability_real *= 100

    probability_fake = 100 - probability_real

    # =====================================================
    # FINAL RESULT
    # =====================================================

    if prediction == 1:

        result = "REAL NEWS"

        real_probability = probability_real
        fake_probability = probability_fake

    else:

        result = "FAKE NEWS"

        fake_probability = probability_fake
        real_probability = probability_real

    # Confidence = probability of predicted class
    if result == "REAL NEWS":
        confidence = real_probability
    else:
        confidence = fake_probability

    confidence = round(
        confidence,
        2
    )

    fake_probability = round(
        fake_probability,
        2
    )

    real_probability = round(
        real_probability,
        2
    )

    # =====================================================
    # SAVE PREDICTION HISTORY
    # =====================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    new_record = pd.DataFrame([{
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "article": news_title,
        "prediction": result,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": confidence
    }])

    # If history file doesn't exist, create it
    if not os.path.exists(HISTORY_FILE):

        new_record.to_csv(
            HISTORY_FILE,
            index=False
        )

    else:

        try:

            existing = pd.read_csv(
                HISTORY_FILE
            )

            # Check whether old file has correct structure
            required_columns = [
                "time",
                "article",
                "prediction",
                "fake_probability",
                "real_probability",
                "confidence"
            ]

            if not all(
                column in existing.columns
                for column in required_columns
            ):

                # Replace incompatible old history
                new_record.to_csv(
                    HISTORY_FILE,
                    index=False
                )

            else:

                new_record.to_csv(
                    HISTORY_FILE,
                    mode="a",
                    header=False,
                    index=False
                )

        except Exception:

            # Replace corrupted history file
            new_record.to_csv(
                HISTORY_FILE,
                index=False
            )

    # =====================================================
    # UPDATE ANALYTICS
    # =====================================================

    analytics = get_live_analytics()

    # =====================================================
    # RETURN TO SAME INTERFACE
    # =====================================================

    return render_template(
        "index.html",

        # Result expected by your HTML
        result=result,

        # Prediction values
        confidence=confidence,
        fake_probability=fake_probability,
        real_probability=real_probability,

        # Keep entered text
        news_title=news_title,
        news_text=news_text,

        # Analytics
        analytics=analytics
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )