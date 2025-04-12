import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import os
import random

def main():
    # --- Cấu hình ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 2  # violence vs non_violence

    # --- Transform ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Load dataset ---
    full_dataset = datasets.ImageFolder('dataset/train', transform=transform)
    class_to_idx = full_dataset.class_to_idx  # ví dụ: {'non_violence': 0, 'violence': 1}
    print("Class to index mapping:", class_to_idx)

    # --- Phân chia 10% mỗi class cho validation ---
    indices_by_class = {cls: [] for cls in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset.samples):
        indices_by_class[label].append(idx)

    # Shuffle để lấy ngẫu nhiên
    random.seed(42)
    for cls in indices_by_class:
        random.shuffle(indices_by_class[cls])

    val_indices = []
    train_indices = []

    for cls in range(num_classes):
        cls_indices = indices_by_class[cls]
        val_count = int(0.1 * len(cls_indices))  # 10% mỗi class
        val_indices += cls_indices[:val_count]
        train_indices += cls_indices[val_count:]

    # Tạo Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load mô hình ---
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)

    if os.path.exists('checkpoints/vit_small_violence.pth'):
        model.load_state_dict(torch.load('checkpoints/vit_small_violence.pth'))
        print("Loaded pretrained weights.")
    else:
        print("Training from scratch.")

    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Train ---
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

        # --- Validation ---
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

    # --- Save model ---
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vit_small_violence.pth')
    print("Model saved to checkpoints/vit_small_violence1.pth")

if __name__ == '__main__':
    main()
