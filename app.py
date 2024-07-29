from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Path to your model
MODEL_PATH = 'model85.h5'

# Class labels
class_labels = {
    0: "cloudy",
    1: "desert",
    2: "green area",
    3: "water"
}

# Load the model once when the app starts
model = load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(img, target_size=(255, 255)):
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the class of an image
def predict_image(img):
    preprocessed_image = preprocess_image(img)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_labels.get(predicted_class_index, "Unknown")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            img = Image.open(file.stream)
            result = predict_image(img)
            return render_template('results.html', result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
