from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("model/nb_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review = ""

    if request.method == "POST":
        review = request.form["review"]
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)

        prediction = "Positive Review 😊" if result[0] == 1 else "Negative Review 😞"

    return render_template("index.html", prediction=prediction, review=review)

if __name__ == "__main__":
    app.run(debug=True)
