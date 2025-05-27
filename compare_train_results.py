import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đường dẫn file kết quả
csv_files = [
    'train_results.csv',
    'logs/train_results_finetune.csv',
    'logs/train_results_cnns_vit.csv'
]

# Đọc dữ liệu từ tất cả các file có tồn tại
dfs = []
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)

if not dfs:
    print('Không tìm thấy file kết quả train nào!')
    exit(1)

# Gộp dữ liệu
all_results = pd.concat(dfs, ignore_index=True)

# Chỉ lấy bản ghi mới nhất cho mỗi model (tránh trùng lặp nếu có nhiều lần train)
all_results = all_results.sort_values('datetime').groupby('model').tail(1)

# Vẽ biểu đồ cột so sánh
metrics = ['train_acc', 'val_acc', 'train_loss']
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.barplot(x='model', y=metric, data=all_results, palette='Set2')
    plt.title(metric)
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    for index, row in all_results.iterrows():
        plt.text(index, row[metric], f'{row[metric]:.3f}', ha='center', va='bottom', fontsize=10)
plt.suptitle('So sánh các mô hình (theo kết quả thực tế)', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('logs/compare_all_models.png', dpi=300, bbox_inches='tight')
plt.show()
print('Đã lưu biểu đồ so sánh vào logs/compare_all_models.png') 