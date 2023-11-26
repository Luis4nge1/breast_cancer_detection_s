from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from werkzeug.utils import secure_filename
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Cargar el modelo .h5
modelo = load_model('modelo_bc1_cnn.h5')
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

loaded_model_lr = joblib.load('modelo_logistic_regression.joblib')
loaded_model_rf = joblib.load('modelo_random_forest.joblib')
loaded_model_dt = joblib.load('modelo_decision_tree.joblib')
loaded_model_svc = joblib.load('modelo_svc.joblib')

class_names = ['benign', 'malignant']

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/redirect_index')
def redirect_index():
    return redirect(url_for('index'))

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
  if request.method == 'POST':
        if 'imagen' not in request.files:
            return render_template('predict_image.html', mensaje='No se proporcionó ninguna imagen')

        imagen = request.files['imagen']

        if imagen.filename == '':
            return render_template('predict_image.html', mensaje='No se seleccionó ninguna imagen')

        if imagen and allowed_file(imagen.filename):

            # Guardar la imagen en el servidor
            filename = secure_filename(imagen.filename)
            ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            imagen.save(ruta_imagen)

            mensaje = "Hola"

            data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)
            image = Image.open(imagen)
            size = (180, 180)
            image = ImageOps.exif_transpose(image)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            prediction = modelo.predict(data)
            print(prediction)

            label = np.argmax(prediction)

            if label == 0:
              mensaje ="The image is most likely benign"
            else:
              mensaje = "The image is most likely malignant"

            img = tf.keras.utils.load_img(ruta_imagen, target_size=(180, 180))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = modelo.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            result = {
                'class': class_names[np.argmax(score)],
                'confidence': 100 * np.max(score)
            }

            tipomensaje = ""

            if class_names[np.argmax(score)] == 'benign':
               tipomensaje = "benign"
               mensaje = "The image is most likely {} with a {:.5f} percent confidence.".format(tipomensaje, 100 * np.max(score))
               return render_template('predict_image.html', mensajepositivo=mensaje, filename=filename)
            else:
               tipomensaje = "malignant"
               mensaje = "The image is most likely {} with a {:.5f} percent confidence.".format(tipomensaje, 100 * np.max(score))
               return render_template('predict_image.html', mensajenegativo=mensaje, filename=filename)
  return render_template('predict_image.html')

@app.route('/procesar', methods=['GET', 'POST'])
def procesar():
  if request.method == 'POST':
        radius_mean = float(request.form['radius'])
        texture_mean = float(request.form['texture'])
        perimeter_mean = float(request.form['perimeter'])
        area_mean = float(request.form['area'])
        smoothness_mean = float(request.form['smoothness'])
        compactness_mean = float(request.form['compactness'])
        concavity_mean = float(request.form['concavity'])
        concavepoints_mean = float(request.form['concavepoints'])
        symmetry_mean = float(request.form['symmetry'])
        fractaldimension_mean = float(request.form['fractaldimension'])

        new_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, 
                              smoothness_mean, compactness_mean, concavity_mean, concavepoints_mean, symmetry_mean, fractaldimension_mean]])

        new_predictions_lr = loaded_model_lr.predict(new_data)
        new_predictions_rf = loaded_model_rf.predict(new_data)
        new_predictions_dt = loaded_model_dt.predict(new_data)
        new_predictions_svc = loaded_model_svc.predict(new_data)

        resultado = [tipo_prediction(new_predictions_lr[0]), tipo_prediction(new_predictions_rf[0]), tipo_prediction(new_predictions_dt[0]), tipo_prediction(new_predictions_svc[0])]

        print(f'Predictions for new data: {new_predictions_lr}')

        return render_template('predict_data.html', prediction=resultado)
        
  return render_template('predict_data.html')

def tipo_prediction(resultado):
  mensaje = ""
  if resultado == 1:
    mensaje = "Malignant"
  else:
     mensaje = "Benign"
  return mensaje
      

# Función para verificar la extensión del archivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


if __name__ == '__main__':
  app.run(port=5000)
