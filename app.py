from flask import Flask, redirect, render_template, request
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from PIL import Image
import os
import torch.nn as nn

app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=4)
model.load_state_dict(torch.load("Resnet18_corn_leaf.h5", map_location=torch.device('cpu')))
model.eval()

def predict_image(image_path, model, class_names):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    
    probabilities = torch.softmax(prediction, dim=1)
    predicted_class_index = torch.argmax(probabilities).item()
    confidence_score = probabilities[0][predicted_class_index].item()
    confidence_score = round(confidence_score, 4) * 100
    
    return class_names[predicted_class_index], confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    
    class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', 'images', file.filename)
            file.save(file_path)
            
            predicted_class, confidence_score = predict_image(file_path, model, class_names)
            print(predicted_class, confidence_score)
                
            return render_template('index.html', predicted_class=predicted_class, confidence_score=confidence_score, image_path=file_path)
    return render_template('index.html', predictions=None, image_path=None)


if __name__ == '__main__':
    app.run()
