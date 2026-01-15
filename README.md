

https://github.com/user-attachments/assets/15f4e42b-ad3c-4051-9453-49d2d843dd89





# 🎬 Movie Review Sentiment Analysis

A complete **end-to-end Machine Learning web application** that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using **Naive Bayes**.  
The project includes data preprocessing, model training, evaluation, and a cinematic Flask-based web interface.

---

##  Project Overview

Movie reviews contain rich textual information that reflects audience sentiment.  
This project builds a **text classification system** that:

- Takes raw movie reviews as input
- Processes and converts text into numerical features using **TF-IDF**
- Predicts sentiment using a **Multinomial Naive Bayes** model
- Displays results through a **modern HTML/CSS web interface**

---

##  Machine Learning Pipeline

1. **Dataset Selection**
   - IMDb Dataset of 50K Movie Reviews
   - Binary labels: Positive / Negative

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation & numbers
   - Stopword removal using NLTK

3. **Feature Engineering**
   - TF-IDF Vectorization (Top 5000 features)

4. **Model Training**
   - Multinomial Naive Bayes
   - 80% Training / 20% Testing split

5. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix visualization

6. **Deployment**
   - Flask backend
   - HTML + CSS frontend (cinematic dark theme)

---

##  Model Performance

- Achieves **high accuracy (~85–90%)**
- Balanced precision and recall
- Effective handling of real-world movie reviews

---

##  Web Application Features

- Clean, movie-themed UI
- Large review input area
- Instant sentiment prediction
- Emoji-based feedback for better UX

---

## 📁 Project Structure

Text-Classification_project/
│
├── data/
│ └── IMDB Dataset.csv
│
├── model/
│ ├── nb_model.pkl
│ └── tfidf.pkl
│
├── templates/
│ └── index.html
│
├── static/
│ ├── style.css
│ └── bg.jpg
│
├── sentiment_analysis.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


---

##  How to Run the Project

### 1️ Create & activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python sentiment_analysis.py

python app.py

http://127.0.0.1:5000





