from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import joblib
from skimage import feature

from sklearn.preprocessing import StandardScaler

app = Flask(__name__,static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
model_path = 'C:/Users/HIMAJA/vscodeprojects/Medilab/spiral_rf_model_four.pkl'
model_path_two='C:/Users/HIMAJA/vscodeprojects/Medilab/wave_rf_model.pkl'
model = joblib.load(model_path)
model_two=joblib.load(model_path_two)

model_path_three = 'C:/Users/HIMAJA/vscodeprojects/Medilab/svm_two.pkl'

# try:
#     model_three = tf.keras.models.load_model(model_path_three)
#     print("TensorFlow model loaded successfully!")
# except Exception as e:
#     print("Error loading TensorFlow model:", e)

model_three=joblib.load(model_path_three)




def quantify_image(image):
    # Placeholder for actual image quantification logic
    
    # Check if the input image has only one channel
    if len(image.shape) == 2:
        # If it has only one channel, it's already grayscale
        gray = image
    else:
        # If it has multiple channels, convert it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply other preprocessing steps and feature extraction
    gray = cv2.resize(gray, (200, 200))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    features = feature.hog(thresh, orientations=9, pixels_per_cell=(10, 10),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return features


# def test_prediction(model, file_path):
#     image = cv2.imread(file_path)
#     output = image.copy()
#     output = cv2.resize(output, (128, 128))
#     # Pre-process the image
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.resize(image, (200, 200))
#     _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     # Quantify the image and make predictions
#     features = quantify_image(image)
#     preds = model.predict([features])
#     label = "Parkinson's" if preds[0] else "Healthy"
#     # Draw the label on the image
#     color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
#     cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     output_path = os.path.join('static', secure_filename(file_path))
#     cv2.imwrite(output_path, output)
#     return output_path, label

def test_prediction(model, file_path):
    image = cv2.imread(file_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    # Pre-process the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Quantify the image and make predictions
    features = quantify_image(image)
    preds = model.predict([features])
    label = "Parkinson Disease" if preds[0] else "Healthy patient"
    # Draw the label on the image
    color = (0, 255, 0) if label == "Healthy patient" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Generate a secure filename for the result image
    result_filename = secure_filename(file_path)
    # Save the result image to the static folder
    result_path = os.path.join('static', result_filename)
    cv2.imwrite(result_path, output)
    
    # Return the relative path to the result image
    return result_filename, label




def preprocess_input(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data



@app.route('/')
def index():
    # Redirect to index2.html
    return render_template("index.html")

@app.route('/index2')
def index2():
    # Redirect to indexnew.html
    return render_template("index2.html")

@app.route('/indexnew')
def indexnew():
    # Render indexnew.html
    return render_template('indexnew.html')

@app.route('/parkinsons')
def parkinsons():
    return render_template("parkinsons.html")

@app.route('/indexwave')
def indexwave():
    return render_template("index_wave.html")

@app.route('/parkcause')
def parkcause():
    return render_template("parkinsoncause.html")

@app.route('/indexvoice')
def voice():
    return render_template("index-voice.html")


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result_image, prediction = test_prediction(model, file_path)
        image_filename = os.path.join('uploads', filename)
        return render_template('result.html', prediction=prediction, image_filename=result_image)
    

@app.route('/uploadtwo', methods=['POST'])
def uploadtwo():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result_image, prediction = test_prediction(model_two, file_path)
        image_filename = os.path.join('uploads', filename)
        return render_template('resultwave.html', prediction=prediction, image_filename=result_image)
    

@app.route('/detect', methods=['POST'])
def detect():
    # input_data = []
    # for key in request.form:
    #     input_data.append(float(request.form[key]))

    # # Preprocess the input data
    # input_data = np.array(input_data).reshape(1, -1)

    # # Make prediction
    # pred = model_three.predict(input_data)
    # # Assuming 1 is the positive class and 0 is the negative class
    # class_prediction = pred[0]
    # print(class_prediction)

    # # Determine the result message
    # if class_prediction == 0:
    #     result = "Parkinson's disease not detected"
    # else:
    #     result = "Parkinson's disease detected"

    # return render_template('resultvoice.html', result=result)

    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            mdvp_fo=float(request.form['mdvp_fo'])
            mdvp_fhi=float(request.form['mdvp_fhi'])
            mdvp_flo=float(request.form['mdvp_flo'])
            mdvp_jitper=float(request.form['mdvp_jitper'])
            mdvp_jitabs=float(request.form['mdvp_jitabs'])
            mdvp_rap=float(request.form['mdvp_rap'])
            mdvp_ppq=float(request.form['mdvp_ppq'])
            jitter_ddp=float(request.form['jitter_ddp'])
            mdvp_shim=float(request.form['mdvp_shim'])
            mdvp_shim_db=float(request.form['mdvp_shim_db'])
            shimm_apq3=float(request.form['shimm_apq3'])
            shimm_apq5=float(request.form['shimm_apq5'])
            mdvp_apq=float(request.form['mdvp_apq'])
            shimm_dda=float(request.form['shimm_dda'])
            nhr=float(request.form['nhr'])
            hnr=float(request.form['hnr'])
            rpde=float(request.form['rpde'])
            dfa=float(request.form['dfa'])
            spread1=float(request.form['spread1'])
            spread2=float(request.form['spread2'])
            d2=float(request.form['d2'])
            ppe=float(request.form['ppe'])

            input_data = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitper, mdvp_jitabs, mdvp_rap, mdvp_ppq, jitter_ddp,
                 mdvp_shim, mdvp_shim_db, shimm_apq3, shimm_apq5, mdvp_apq, shimm_dda, nhr, hnr, rpde, dfa,
                 spread1, spread2, d2, ppe]
            
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            
            prediction = model_three.predict(input_data_reshaped)
            if prediction == 1:
                pred = "You have Parkinson's Disease. Please consult a specialist."
            else:
                pred = "You are Healthy Person."
            # showing the prediction results in a UI
            return render_template('resultvoice.html',prediction=pred)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index2.html')


if __name__ == '__main__':
    app.run(debug=True)




if __name__ == '__main__':
    app.run(debug=True)
