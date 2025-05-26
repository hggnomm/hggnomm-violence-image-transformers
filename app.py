import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from flask import Flask, request, render_template
import os

# Cấu hình
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = 2
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('model/vit_small_violence.pth', map_location=device))
model.to(device)
model.eval()

# Transform ảnh giống lúc train
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hàm predict
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = 'Bạo lực' if predicted.item() == 1 else 'Không bạo lực'
        percentage = confidence.item() * 100
        return label, round(percentage, 2)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="Không tìm thấy tệp.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="Chưa chọn tệp.")

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        label, percentage = predict_image(path)
        prediction = label
        confidence = percentage
        image_url = path

    return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
