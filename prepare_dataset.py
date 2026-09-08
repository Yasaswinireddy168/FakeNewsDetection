import pandas as pd

# Load the two datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

# Add labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
news_data = pd.concat([fake_news, real_news], ignore_index=True)

# Remove duplicate rows
news_data = news_data.drop_duplicates()

# Shuffle the dataset
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
news_data.to_csv("dataset/news.csv", index=False)

# Display information
print("Dataset prepared successfully!")
print("Final dataset shape:", news_data.shape)

print("\nClass distribution:")
print(news_data["label"].value_counts())

print("\nFirst 5 rows:")
print(news_data.head())
print("\n--- SAMPLE ARTICLE TEXT ---")

print("\nFake article:")
print(fake_news["text"].iloc[0])

print("\nReal article:")
print(real_news["text"].iloc[0])