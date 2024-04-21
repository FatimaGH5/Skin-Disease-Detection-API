import io
import json
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load your own model (replace with your model loading logic)
model = models.mobilenet_v2(pretrained=True)

# Load the model's state dictionary
state_dict = torch.load("model2", map_location=torch.device('cpu'))

# Instantiate the model
model = models.mobilenet_v2()
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.7630392, 0.5456477, 0.57004845],  # Standard normalization for ImageNet model input
                             [0.1409286, 0.15261266, 0.16997074])
    ]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)  # Open the image file
    timg = my_transforms(image)  # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)  # PyTorch models expect batched input; create a batch of 1
    return timg

# Get a prediction
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)  # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)  # Extract the most likely class
    prediction = y_hat.item()  # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    class_labels = {
        0: "Acne and Rosacea",
        1: "Actinic keratoses",
        2: "Atopic Dermatitis",
        3: "Basal cell carcinoma",
        4: "Benign keratosis-like lesions",
        5: "Dermatofibroma",
        6: "Light Diseases and Disorders of Pigmentation",
        7: "Melanocytic nevi",
        8: "Seborrheic Keratoses and other Benign Tumors",
        9: "Vascular lesions",
        10: "Warts Molluscum and other Viral Infections",
        11: "melanoma"
    }

    class_name = class_labels.get(prediction_idx, "Unknown")
    return prediction_idx, class_name

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                input_tensor = transform_image(file)
                prediction_idx = get_prediction(input_tensor)
                class_id, class_name = render_prediction(prediction_idx)
                return jsonify({'class_id': class_id, 'class_name': class_name}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'No file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)



