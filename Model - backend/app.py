from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model architecture and weights
with open("model_architecture.json", "r") as json_file:
    model_architecture = json_file.read()

model = model_from_json(model_architecture)
model.load_weights("model_weights.weights.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Load max_length from the saved file
with open("max_length.txt", "r") as file:
    max_length = int(file.read())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_text = data.get("text")
    
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)

    # Make prediction
    try:
        prediction = model.predict(padded_input_sequence)
        predicted_label_index = np.argmax(prediction[0])
        predicted_label = label_encoder.inverse_transform([predicted_label_index])

        return jsonify({"emotion": predicted_label[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)




