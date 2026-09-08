import pandas as pd

fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

print("FAKE NEWS DATASET")
print(fake_news.head())

print("\nREAL NEWS DATASET")
print(real_news.head())

print("\nFake dataset shape:", fake_news.shape)
print("Real dataset shape:", real_news.shape)

print("\nFake dataset columns:")
print(fake_news.columns)

print("\nReal dataset columns:")
print(real_news.columns)
print("\n--- MISSING VALUES ---")

print("Fake dataset missing values:")
print(fake_news.isnull().sum())

print("\nReal dataset missing values:")
print(real_news.isnull().sum())


print("\n--- DUPLICATES ---")

print("Duplicate rows in Fake dataset:", fake_news.duplicated().sum())
print("Duplicate rows in Real dataset:", real_news.duplicated().sum())
print("\n--- SAMPLE ARTICLE TEXT ---")

print("\nFake article:")
print(fake_news["text"].iloc[0])

print("\nReal article:")
print(real_news["text"].iloc[0])