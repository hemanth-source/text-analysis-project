import os
import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords (only first time)
nltk.download('stopwords')

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("data/IMDB Dataset.csv")

# 2. Text Preprocessing
print("Preprocessing text...")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# Encode labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3. Train-Test Split
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Extraction (TF-IDF)
print("Extracting features...")
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Train Naive Bayes Model
print("Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Predictions
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation Results")
print("-------------------------")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.show()

# 9. Custom Review Prediction
print("\nCustom Review Test")
print("------------------")
sample_review = "i did not like the movie it was boring and too long"
clean_sample = clean_text(sample_review)
sample_vector = tfidf.transform([clean_sample])
prediction = model.predict(sample_vector)

print("Review:", sample_review)
print("Predicted Sentiment:", "Positive" if prediction[0] == 1 else "Negative")

# 10. Save the trained model and vectorizer
print("\nSaving model and vectorizer...")
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/nb_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("Model and TF-IDF vectorizer saved successfully!")
