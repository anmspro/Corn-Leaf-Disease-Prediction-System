from flask import Flask, redirect, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model_vgg16 = load_model('Vgg16_corn_leaf.h5')
model_vgg19 = load_model('Vgg19_corn_leaf.h5')
model_alexnet = load_model('Alexnet_corn_leaf.h5')
model_mobilenet = load_model('MobileNetv2_corn_leaf.h5')
model_resnet = load_model('Resnet18_corn_leaf.h5')
model_inceptionv3 = load_model('SqueezeNet_corn_leaf.h5')

def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(np.array(prediction))
    return prediction, predicted_class_index

def predict_image75(image_path, model):
    img = image.load_img(image_path, target_size=(75, 75))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(np.array(prediction))
    return prediction, predicted_class_index

@app.route('/', methods=['GET', 'POST'])
def index():
    models = {
        'VGG16': model_vgg16,
        'VGG19': model_vgg19,
        'AlexNet': model_alexnet,
        'MobileNetV2': model_mobilenet,
        'ResNet18': model_resnet,
        'InceptionV3': model_inceptionv3
    }
    
    class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']
    predictions = {}

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', 'images', file.filename)
            file.save(file_path)
            
            for model_name, model in models.items():
                if model_name != 'InceptionV3':
                    prediction, predicted_class_index = predict_image(file_path, model)
                    predicted_class = class_names[predicted_class_index]
                    predictions[model_name] = {'class': predicted_class, 'confidence': prediction[0][predicted_class_index]}
                else:
                    prediction, predicted_class_index = predict_image75(file_path, model)
                    predicted_class = class_names[predicted_class_index]
                    predictions[model_name] = {'class': predicted_class, 'confidence': prediction[0][predicted_class_index]}
            
            return render_template('index.html', predictions=predictions, image_path=file_path)
    return render_template('index.html', predictions=None, image_path=None)

if __name__ == '__main__':
    app.run()
