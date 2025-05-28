from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import mysql.connector
import os
import requests

app = Flask(__name__)

MODEL_PATH = 'model.h5'  
model = load_model(MODEL_PATH)

labels = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 
          'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 
          'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon', 
          'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish', 
          'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 
          'Silver Carp', 'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost', 
        user='root',  
        password='111777',  
        database='fish_database' 
    )
    return conn

def fetch_fish_info(species):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT biological_name, status, origin FROM fish_info WHERE species_name = %s"
    cursor.execute(query, (species,))
    result = cursor.fetchone()
    conn.close()
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    image_url = request.form.get('url')
    file_path = None

    if file and file.filename:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

    elif image_url:
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                os.makedirs('uploads', exist_ok=True)
                file_path = os.path.join('uploads', 'downloaded_image.jpg')
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                return "Error: Unable to fetch image from URL", 400
        except Exception as e:
            return f"Error: {str(e)}", 400

    if not file_path:
        return redirect(url_for('index'))

    try:
        image = preprocess_image(file_path)
        predictions = model.predict(image)
        predicted_species = labels[np.argmax(predictions)]

        fish_info = fetch_fish_info(predicted_species)
        if fish_info:
            biological_name, status, origin = fish_info
        else:
            biological_name, status, origin = "Unknown", "Unknown", "Unknown"

        return render_template(
            'result.html',
            species=predicted_species,
            biological_name=biological_name,
            status=status,
            origin=origin,
            confidence=f"{np.max(predictions) * 100:.2f}%"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
