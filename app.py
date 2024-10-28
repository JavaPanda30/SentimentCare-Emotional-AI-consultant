import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from flask import Flask, request, jsonify
import pickle

# Load the fine-tuned model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("roberta_emotion_model")
tokenizer = RobertaTokenizer.from_pretrained("roberta_emotion_model")

# Load the label encoder
with open("roberta_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize Flask app
app = Flask(__name__)

def predict_emotion(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the predicted labels
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    predicted_emotions = label_encoder.inverse_transform(predictions)

    return predicted_emotions

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid input, expected JSON with a 'text' field"}), 400

    # Get input text
    input_text = request.json['text']
    
    # Ensure the input is a list of texts
    if isinstance(input_text, str):
        input_text = [input_text]
    
    # Predict emotions
    predicted_emotions = predict_emotion(input_text)

    # Convert to a Python list for JSON serialization
    predicted_emotions = predicted_emotions.tolist()

    # Return the results
    return jsonify({"predictions": predicted_emotions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
