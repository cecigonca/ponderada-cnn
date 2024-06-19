from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'imagens'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('../modelo/modelo.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '' or not file.content_type.startswith('image/'):
            return redirect(request.url)
        
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)  
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        
        return render_template('index.html', prediction=digit)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
