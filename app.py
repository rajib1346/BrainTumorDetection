from flask import Flask, render_template, request
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Create upload folder if not exists
os.makedirs("static/user_images", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

# Global variables
model2 = None

@app.route('/predictor2', methods=['POST'])  
def predictor2():
    global model2

    # Load model only once
    if model2 is None:
        model2 = load_model("static/models/model_2.h5", compile=False)

    try:
        # Get uploaded image and name
        image = request.files['image']
        name = request.form.get('name')
        
        # Save uploaded image
        image_path = f'static/user_images/{name}.jpg'

        # Open and preprocess the image
        image = Image.open(image.stream)
        image = image.resize((224, 224))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(image_path)

        # Process for prediction
        img = cv2.imread(image_path)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        result = model2.predict(img)
        result = result.squeeze()

        # Determine result
        if result > 0.5:
            prediction_result = "TUMOR DETECTED"
            print("Tumor Detected")
        else:
            prediction_result = "NO TUMOR DETECTED"
            print("No Tumor Detected")

        # ✅ FIXED: Use your modified template
        return render_template('prediction1.html', data=[prediction_result, name])
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = "static/user_images"
    app.run(debug=True, host='0.0.0.0')
