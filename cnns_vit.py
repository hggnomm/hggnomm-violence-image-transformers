import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import os
import random
import csv
from datetime import datetime

# Định nghĩa model kết hợp CNNs + ViT
class CNNs_ViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Thêm một block CNN đơn giản trước ViT
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # trả về 3 kênh để phù hợp ViT
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        # ViT backbone
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.vit(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 2

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder('dataset/train', transform=transform)
    class_to_idx = full_dataset.class_to_idx
    print("Class to index mapping:", class_to_idx)

    # Chia train/val
    indices_by_class = {cls: [] for cls in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset.samples):
        indices_by_class[label].append(idx)
    random.seed(42)
    for cls in indices_by_class:
        random.shuffle(indices_by_class[cls])
    val_indices, train_indices = [], []
    for cls in range(num_classes):
        cls_indices = indices_by_class[cls]
        val_count = int(0.1 * len(cls_indices))
        val_indices += cls_indices[:val_count]
        train_indices += cls_indices[val_count:]

    # --- TEST MODE: Chỉ lấy 30% data để train thử ---
    """
    TODO: Khi chạy với full data, comment đoạn code này lại
    """
    # Lấy 30% data từ tập train
    # train_indices = train_indices[:int(len(train_indices) * 0.3)]
    train_dataset = Subset(full_dataset, train_indices)
    print(f"TEST MODE: Using only 30% of training data ({len(train_indices)} samples)")
    # --- TEST END

    val_dataset = Subset(full_dataset, val_indices)
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = CNNs_ViT(num_classes=num_classes).to(device)
    print("Model: CNNs + ViT")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        print(f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total * 100
        avg_loss = running_loss / total
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cnns_vit.pth')
    print("Model saved to checkpoints/cnns_vit.pth")

    # Lưu kết quả train vào file CSV
    os.makedirs('logs', exist_ok=True)
    result_file = 'logs/train_results_cnns_vit.csv'
    model_name = 'CNNs+ViT-model'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_row = [now, model_name, num_epochs, round(train_acc, 4), round(val_acc, 4), round(avg_loss, 4)]
    header = ['datetime', 'model', 'epochs', 'train_acc', 'val_acc', 'train_loss']
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(result_row)
    print(f"Đã lưu kết quả train vào {result_file} để tiện so sánh các lần train.")

if __name__ == '__main__':
    main() 