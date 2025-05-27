import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from flask import Flask, request, render_template
import os

# ==== Định nghĩa 2 hàm khởi tạo model ====
def get_vit_model(num_classes=2, pretrained=False):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model

def get_cnns_vit_model(num_classes=2, pretrained_vit=True):
    class CNNs_ViT(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU()
            )
            self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained_vit, num_classes=num_classes)
        def forward(self, x):
            x = self.cnn(x)
            x = self.vit(x)
            return x
    return CNNs_ViT(num_classes=num_classes)

# ==== Cấu hình Flask ====
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== CHỌN MODEL TẠI ĐÂY ====
num_classes = 2

# Để dùng ViT gốc:
# model = get_vit_model(num_classes=num_classes, pretrained=False)
# model.load_state_dict(torch.load('checkpoints/vit_small_violence.pth', map_location=device))

# Để dùng CNNs+ViT:
model = get_cnns_vit_model(num_classes=num_classes, pretrained_vit=True)
model.load_state_dict(torch.load('checkpoints/cnns_vit.pth', map_location=device))

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
