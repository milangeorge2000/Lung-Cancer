from distutils.log import debug
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)


saved_model = load_model("denselung.h5") ##loading the model


# model1 = load_model("densemamo6.h5")
# model2 = load_model("dense.h5") ##loading the model

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save("img.jpg")
        
        



        img = image.load_img("img.jpg",target_size=(224,224)) ##loading the image
        img = np.asarray(img) ##converting to an array
        img = img / 255 ##scaling by doing a division of 255
        img = np.expand_dims(img, axis=0) ##expanding the dimensions
  
        
        output = saved_model.predict(img)

        output = np.argmax(output, axis=1) ##Taking the index of the maximum value
        if output[0] == 0 :
        #    result = 'benign'
             return render_template('prediction.html', prediction = 'Benign')
  
        elif output[0] == 1:
        #   result = 'malignant'
            return render_template('prediction.html', prediction = 'Malignant')

        elif output[0] == 2 :
        #    result = 'normal'
             return render_template('prediction.html', prediction = 'Normal')
     
    # return result

if __name__ == '__main__':
    app.run(debug=False)
