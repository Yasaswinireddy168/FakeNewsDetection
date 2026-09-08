import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
# Load the prepared dataset
news_data = pd.read_csv("dataset/news.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", news_data.shape)
# Download English stopwords
nltk.download("stopwords")

# Load English stopwords
stop_words = set(stopwords.words("english"))

print("Number of stopwords:", len(stop_words))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)
# Test the cleaning function
sample_text = news_data["text"].iloc[0]

print("\n--- ORIGINAL TEXT ---")
print(sample_text[:500])

print("\n--- CLEANED TEXT ---")
print(clean_text(sample_text)[:500])
# Clean all news articles
news_data["clean_text"] = news_data["text"].apply(clean_text)

print("\nAll articles cleaned successfully!")
print(news_data[["text", "clean_text"]].head())
# Combine title and cleaned article text
news_data["content"] = news_data["title"] + " " + news_data["clean_text"]

print("\nTitle and article text combined successfully!")
print(news_data["content"].head())
# Save the processed dataset
news_data.to_csv("dataset/processed_news.csv", index=False)

print("\nProcessed dataset saved successfully!")