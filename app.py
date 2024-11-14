import os
from flask import Flask, request,jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import random
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your model
model = load_model('Resnet_model.h5')

# Preprocessing function for the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  # Resize to match your model's input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return img


# Descriptions for cracked tires
cracked_descriptions = [
    "Visible crack running along the tire surface.",
    "Multiple small cracks forming on the tread.",
    "Large crack found near the sidewall of the tire.",
    "Cracks appear due to weathering and aging.",
    "Crack visible at the base of the tire tread.",
    "Tire shows signs of cracking due to improper inflation.",
    "Crack found near the shoulder of the tire.",
    "Tread cracking due to excessive heat exposure.",
    "Longitudinal crack visible on the outer edge.",
    "Minor cracks forming between the treads."
]

# Descriptions for normal/good tires
good_tire_descriptions = [
    "The tire is in excellent condition with no visible defects.",
    "No cracks, bulges, or flat spots; the tire is well-maintained.",
    "The tire appears to be in good shape with normal wear.",
    "No significant wear or damage; tire is fit for continued use.",
    "The tread depth is consistent, and the tire has no visible damage.",
    "No issues found; tire is ready for safe driving.",
    "This tire is in good health with no visible wear.",
    "The tire shows no signs of aging or cracking.",
    "This tire is performing well and has no visible defects.",
    "The tread is intact, and the tire condition is normal."
]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None, image_url=None)

@app.route('/predict', methods=['POST'])
def predict(): 
    file = request.files['file']
    # Ensure filename is secure and save the uploaded file
    filename = secure_filename(file.filename)
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)  # Create 'uploads' folder if it doesn't exist
    
    image_path = os.path.join(upload_folder, filename)
    file.save(image_path)

    try:
        # Preprocess the image and get predictions
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        class_id = np.argmax(predictions[0])

        # Determine the analysis based on the class_id
        if class_id == 0:  # Tire is cracked
            description = random.choice(cracked_descriptions)
            tire_life = random.randint(1,3)
            response = {
                "status": "Cracked",
                "description": description,
                "life_expectancy": f"{tire_life} months"
            }
        elif class_id == 1:  # Tire is in good condition
            description = random.choice(good_tire_descriptions)
            tire_life = random.randint(12, 18)
            response = {
                "status": "Good Condition",
                "description": description,
                "life_expectancy": f"{tire_life} months"
            }
        else:
            response = {
                "error": "Unexpected prediction class."
            }

        # Clean up the uploaded file after processing
        os.remove(image_path)

        # Return the analysis as a JSON response
        return jsonify(response)

    except Exception as e:
        # In case of error, ensure the uploaded file is removed
        os.remove(image_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
