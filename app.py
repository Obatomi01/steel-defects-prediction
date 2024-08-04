from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('steel_defects_augmented.h5')

# Define the class names
class_names = ['bumped', 'rusted', 'scratched', 'smooth', 'stained']  # Replace with your actual class names

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] if this was done during training
    return img_array

@app.route('/')
def home():
    return jsonify({'message': 'Fetch successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    # Save the uploaded file to a temporary location
    file_path = 'temp.jpg'
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]

    return jsonify({'predicted_class': f'The steel is {predicted_class_name}'})

if __name__ == '__main__':
    app.run(debug=True)

# def immediate_prediction():
#     img_array = preprocess_image('./flowers/dandelion/2076141453_c63801962a_m.jpg')

#     # Make prediction
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)
#     predicted_class_name = class_names[predicted_class[0]]
    
#     print(predicted_class_name)
#     return jsonify({'predicted_class': predicted_class_name})

# immediate_prediction()
