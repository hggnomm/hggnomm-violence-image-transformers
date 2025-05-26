# Violence Detection using Vision Transformer

Dự án phát hiện bạo lực trong hình ảnh sử dụng mô hình Vision Transformer (ViT). Dự án bao gồm các phần training model, fine-tuning và web application để demo.

## Cấu trúc dự án

```
.
├── app.py              # Web application
├── main.py            # Training script
├── finetune.py        # Fine-tuning script
├── templates/         # HTML templates
├── static/           # Static files (CSS, JS, uploads)
├── checkpoints/      # Saved models
├── dataset/          # Training data
│   ├── train/       # Main training data
│   └── new_data/    # Data for fine-tuning
└── model/           # Model files
```

## Chi tiết các thành phần

### 1. Training Model (`main.py`)

Script chính để training model Vision Transformer:
- Sử dụng mô hình ViT với kiến trúc `vit_small_patch16_224`
- Dataset được chia thành 2 class: bạo lực và không bạo lực
- Các thông số training:
  - Batch size: 16
  - Số epochs: 10
  - Learning rate: 1e-4
  - Optimizer: AdamW
  - Loss function: CrossEntropyLoss
- Dataset được chia theo tỷ lệ 90% training và 10% validation
- Model được lưu vào thư mục `checkpoints/vit_small_violence.pth`

### 2. Fine-tuning Model (`finetune.py`)

Script để fine-tune model trên dataset mới:
- Sử dụng model đã train để fine-tune trên dataset mới
- Các thông số fine-tuning:
  - Batch size: 16
  - Số epochs: 5
  - Learning rate: 1e-5 (thấp hơn để fine-tune)
- Model sau khi fine-tune được lưu vào `checkpoints/vit_small_violence_finetuned.pth`

### 3. Web Application (`app.py`)

Web interface để demo model:
- Sử dụng Flask để tạo web interface
- Cho phép người dùng upload ảnh để kiểm tra
- Sử dụng model đã train để dự đoán
- Trả về kết quả: nhãn (bạo lực/không bạo lực) và độ tin cậy (confidence)

## Quy trình xử lý ảnh

Ảnh đầu vào được xử lý qua các bước:
1. Resize về kích thước 224x224
2. Áp dụng các biến đổi:
   - Resize
   - Random horizontal flip (khi training)
   - Convert to tensor
   - Normalize với mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

## Công nghệ sử dụng

- PyTorch cho deep learning
- Vision Transformer (ViT) làm backbone model
- Flask cho web application
- CUDA support cho GPU acceleration

## Cài đặt và sử dụng

1. Cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision timm flask pillow
```

2. Training model:
```bash
python main.py
```

3. Fine-tuning model (nếu cần):
```bash
python finetune.py
```

4. Chạy web application:
```bash
python app.py
```

## Lưu ý

- Đảm bảo có đủ dữ liệu training trong thư mục `dataset/train`
- Dữ liệu fine-tuning (nếu cần) đặt trong thư mục `dataset/new_data`
- Model sẽ được lưu trong thư mục `checkpoints`
- Web application sẽ chạy ở địa chỉ mặc định: http://localhost:5000 