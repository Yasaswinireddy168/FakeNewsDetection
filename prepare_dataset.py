import pandas as pd
import re

# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

# Assign labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine title and article text
fake_news["content"] = (
    fake_news["title"].fillna("") + " " +
    fake_news["text"].fillna("")
)

real_news["content"] = (
    real_news["title"].fillna("") + " " +
    real_news["text"].fillna("")
)

# Remove obvious source cues such as Reuters
def remove_source_cues(text):
    text = str(text)

    text = re.sub(
        r'\breuters\b',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


fake_news["content"] = fake_news["content"].apply(
    remove_source_cues
)

real_news["content"] = real_news["content"].apply(
    remove_source_cues
)

# Keep only required columns
fake_news = fake_news[["content", "label"]]
real_news = real_news[["content", "label"]]

# Combine
news_data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

# Remove duplicate articles
news_data = news_data.drop_duplicates(
    subset=["content"]
)

# Shuffle
news_data = news_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
news_data.to_csv(
    "dataset/news.csv",
    index=False
)

print("Dataset prepared successfully!")
print("Final dataset shape:", news_data.shape)

print("\nClass distribution:")
print(news_data["label"].value_counts())

print("\nColumns:")
print(news_data.columns.tolist())

print("\nSample:")
print(news_data.head())