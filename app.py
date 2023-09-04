from flask import Flask, redirect, render_template, request
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from PIL import Image
import os
import torch.nn as nn

app = Flask(__name__)
# model_vgg16 = torch.load('Vgg16_corn_leaf.pth', map_location=torch.device('cpu'))
# model_vgg19 = torch.load('Vgg19_corn_leaf.pth', map_location=torch.device('cpu'))
# model_alexnet = torch.load('Alexnet_corn_leaf.pth', map_location=torch.device('cpu'))
# model_mobilenet = torch.load('MobileNetv2_corn_leaf.pth', map_location=torch.device('cpu'))
# model_resnet = torch.load('Resnet18_corn_leaf.pth', map_location=torch.device('cpu'))
# model_SqueezeNet = torch.load('SqueezeNet_corn_leaf.pth', map_location=torch.device('cpu'))

# model_vgg16.eval()
# model_vgg19.eval()
# model_alexnet.eval()
# model_mobilenet.eval()
# model_resnet.eval()
# model_SqueezeNet.eval()


model = models.vgg16(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=4)
model.load_state_dict(torch.load("Vgg16_corn_leaf.h5", map_location=torch.device('cpu')))
model.eval()

def predict_image(image_path, model):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    predicted_class_index = torch.argmax(prediction).item()
    return prediction, predicted_class_index

@app.route('/', methods=['GET', 'POST'])
def index():
    models = {
        # 'VGG16': model_vgg16,
        # 'VGG19': model_vgg19,
        # 'AlexNet': model_alexnet,
        # 'MobileNetV2': model_mobilenet,
        # 'ResNet18': model_resnet,
        # 'SqueezeNet': model_SqueezeNet
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
            
            # for model_name, model in models.items():
            prediction, predicted_class_index = predict_image(file_path, model)
            predicted_class = class_names[predicted_class_index]
            print(predicted_class)
            # predictions[model_name] = {'class': predicted_class, 'confidence': prediction[0][predicted_class_index].item()}
                
            return render_template('index.html', predicted_class=predicted_class, image_path=file_path)
    return render_template('index.html', predictions=None, image_path=None)

if __name__ == '__main__':
    app.run()
