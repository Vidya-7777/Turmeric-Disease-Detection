from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from turmeric_utils import load_custom_dataset, train_classifier, predict_image, suggest_remedy, segment_disease
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load dataset and model
dataset_path = "Turmeric Plant Disease"
images, labels, label_names = load_custom_dataset(dataset_path)
simple_model = train_classifier(images, labels)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    predicted = predict_image(img_path, simple_model, label_names)
    remedy = suggest_remedy(predicted)

    segmented = segment_disease(img_path)
    segmented_name = f"segmented_{filename}"
    segmented_path = os.path.join("static", segmented_name)
    cv2.imwrite(segmented_path, segmented)

    return jsonify({
        "prediction": predicted,
        "remedy": remedy,
        "segmented_path": segmented_name
    })

if __name__ == "__main__":
    app.run(debug=True)
