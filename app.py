import flask
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import base64
import io
from tensorflow import keras

# Create a Flask app
app = Flask(__name__)

# Load the prediction model
model = keras.models.load_model("../vgg16_with_data_aug.keras")


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("website.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = flask.request.files.get('image')

    # Load the image into a PIL Image object
    image = Image.open(image_data).convert('RGB')
    # image.show() => werkt
    shape = (64, 64)
    image = image.resize(shape)

    # Convert the image to a numpy array
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # Perform the prediction
    prediction = model.predict(image)
    # Determine the class with the highest probability
    predicted_class = np.argmax(prediction)
    # Map the integer class label to a string label
    categories = ['Picasso','Rubens','VanGogh']
    predicted_label = categories[predicted_class]

    # Return the prediction as a JSON response
    return render_template('website.html', pred=str(predicted_label))


if __name__ == '__main__':
    app.run(debug=True)
