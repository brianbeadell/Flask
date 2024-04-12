from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch
import os

app = Flask(__name__)

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)  # Ensure num_classes matches your output layer
model.load_state_dict(torch.load('models/my_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    classes = ['Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_healthy']
    return classes[predicted.item()]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/my_projects')
def my_projects():
    return render_template('my_projects.html')

@app.route('/capstone_project')
def capstone_project():
    return render_template('capstone_project.html')

@app.route('/resources_and_references')
def resources_and_references():
    return render_template('resources_and_references.html')

@app.route('/apple_disease_identifier')
def apple_disease_identifier():
    return render_template('apple_disease_identifier.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_disease(filepath)
        # Updating my template to show the prediction result
        return render_template('prediction.html', prediction=prediction, image_path=filepath)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
