# Backend (Python) with Flask

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Load the data from the Excel file
data = pd.read_csv('labeled_data.csv')

# Preprocess the text data
tweets = data['tweet'].values
labels = data['class'].values

# Tokenizer for preprocessing text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(tweets)

# Pad sequences to a fixed length
max_sequence_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Load the trained model
model_path = 'trained_model.h5'
model = load_model(model_path)

# API endpoint to handle text inputs and return predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json  # Receive JSON data with 'text' key containing the input text
    text = data['text']

    # Tokenize and preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    # Make predictions using the loaded model
    prediction = model.predict(padded_sequence)
    predicted_class = int(prediction.argmax())  # Convert numpy int to Python int
    # Process the prediction results
    class_names = ['Hateful Content', 'Offensive Content', 'Neither']
    predicted_label = class_names[predicted_class]
    print(text, prediction);
    # Convert the prediction result to a JSON-serializable format
    response = {'prediction': predicted_label}

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8000)  # Start the Flask app on port 5000
