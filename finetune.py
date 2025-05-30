import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import os
import random
import csv
from datetime import datetime
import time  # Add time module for tracking

def prepare_dataloaders(data_path, transform, batch_size, num_classes=2):
    dataset = datasets.ImageFolder(data_path, transform=transform)

    # Lấy chỉ số mẫu theo từng class
    indices_by_class = {cls: [] for cls in range(num_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(idx)

    # Shuffle và chia 10% mỗi class cho validation
    val_indices, train_indices = [], []
    for cls in range(num_classes):
        cls_indices = indices_by_class[cls]
        random.shuffle(cls_indices)
        val_count = int(0.1 * len(cls_indices))
        val_indices += cls_indices[:val_count]
        train_indices += cls_indices[val_count:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Cấu hình
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-5  # lr thấp hơn để fine-tune
    num_classes = 2

    # Biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataloader từ new_data
    train_loader, val_loader = prepare_dataloaders("dataset/new_data", transform, batch_size, num_classes)

    # In ra số lượng ảnh trong folder new_data
    total_imgs = sum([len(files) for r, d, files in os.walk("dataset/new_data")])
    num_train = len(train_loader.dataset)
    num_val = len(val_loader.dataset)
    print(f"Tổng số ảnh trong dataset/new_data: {total_imgs}")
    print(f"Số ảnh train: {num_train} ({num_train/total_imgs*100:.2f}%) | Số ảnh val: {num_val} ({num_val/total_imgs*100:.2f}%)")

    # --- TEST MODE: Chỉ lấy 30% data để fine-tune thử ---
    # """
    # TODO: Khi chạy với full data, comment đoạn code này lại
    # """
    # train_dataset = train_loader.dataset
    # if hasattr(train_dataset, 'indices'):
    #     train_indices = train_dataset.indices
    # else:
    #     train_indices = list(range(len(train_dataset)))
    # train_indices = train_indices[:int(len(train_indices) * 0.3)]
    # train_dataset = Subset(train_dataset.dataset, train_indices)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # print(f"TEST MODE: Using only 30% of fine-tune training data ({len(train_indices)} samples)")

    # --- test mode end ---


    # Load model ViT + weight cũ
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/vit_small_violence.pth"))
    model = model.to(device)
    print("Loaded pretrained model.")

    # Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Start timing
    start_time = time.time()

    # Fine-tuning
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        print(f"Epoch {epoch+1}/{num_epochs} - Fine-tuning")

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        print(f"Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}%")

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

    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    training_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Lưu mô hình fine-tune
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vit_small_violence_finetuned.pth")
    print("Fine-tuned model saved to checkpoints/vit_small_violence_finetuned.pth")

    # --- Lưu lại kết quả fine-tune vào file CSV để tiện so sánh ---
    os.makedirs('logs', exist_ok=True)
    result_file = 'logs/train_results_finetune.csv'
    model_name = 'ViT-model-more-data'  # Có thể sửa tên này cho các mô hình khác
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_row = [now, model_name, num_epochs, round(train_acc, 4), round(val_acc, 4), round(total_loss/total, 4), training_time]
    header = ['datetime', 'model', 'epochs', 'train_acc', 'val_acc', 'train_loss', 'training_time']
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(result_row)
    print(f"Đã lưu kết quả fine-tune vào {result_file} để tiện so sánh các lần train.")
    print(f"Tổng thời gian training: {training_time}")

if __name__ == "__main__":
    main()
