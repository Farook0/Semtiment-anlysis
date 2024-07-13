import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# Load IMDb dataset for training
def load_training_data(path):
    df = pd.read_csv(path)
    df['review'] = df['review'].apply(data_cleaning)
    return df

# Data Cleaning(preprocessing text)
def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

imdb_df = load_training_data("IMDB Dataset.csv")  

# Split the data
X = imdb_df['review']
y = imdb_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

#Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def comment_anlysis():
    entered_comm = request.form['comment']
    preprocessed_comm = data_cleaning(entered_comm)
    vec_comm = vec.transform([preprocessed_comm])
    predicted_sentiment = model.predict(vec_comm)[0]
    return jsonify({'comment': entered_comm, 'sentiment': predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)