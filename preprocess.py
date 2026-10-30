import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove Reuters/source cue
    text = re.sub(
        r"\breuters\b",
        "",
        text
    )

    # Keep only letters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove stopwords
    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


if __name__ == "__main__":

    # Load prepared dataset
    news_data = pd.read_csv(
        "dataset/news.csv"
    )

    print("Dataset loaded successfully!")
    print(
        "Dataset shape:",
        news_data.shape
    )

    # Clean complete article content
    news_data["clean_text"] = (
        news_data["content"]
        .fillna("")
        .apply(clean_text)
    )

    # Save processed dataset
    news_data.to_csv(
        "dataset/processed_news.csv",
        index=False
    )

    print(
        "Preprocessing completed successfully!"
    )

    print(
        "Processed dataset shape:",
        news_data.shape
    )