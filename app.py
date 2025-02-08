import os
import torch
import whisper
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Whisper AI model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Load trained Decision Tree model
clf = joblib.load("decision_tree_model.joblib")

# Load CountVectorizer
cv = joblib.load("count_vectorizer.joblib")  # Ensure you save this during training

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Transcribe audio using Whisper
        result = whisper_model.transcribe(file_path)
        transcribed_text = result["text"]
        
        # Transform text for prediction
        test_data = cv.transform([transcribed_text]).toarray()
        prediction = clf.predict(test_data)[0]
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "prediction": prediction
        })
    
    return render_template("detect.html")

if __name__ == "__main__":
    app.run(debug=True)
