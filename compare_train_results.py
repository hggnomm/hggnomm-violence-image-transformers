import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Đường dẫn file kết quả
csv_files = [
    'logs/train_results.csv',
    'logs/train_results_finetune.csv',
    'logs/train_results_cnns_vit.csv',
    'logs/train_results_finetune_cnns_vit.csv'
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
metrics = ['train_acc', 'val_acc', 'train_loss', 'training_time']
plt.figure(figsize=(15, 8))
for i, metric in enumerate(metrics):
    plt.subplot(1, 4, i+1)
    if metric == 'training_time':
        # Chuyển đổi thời gian từ HH:MM:SS sang số giây để vẽ biểu đồ
        def convert_time_to_seconds(time_str):
            if isinstance(time_str, str):
                return sum(int(i) * j for i, j in zip(time_str.split(':'), [3600, 60, 1]))
            return time_str  # Return as is if it's already a number
            
        all_results['training_seconds'] = all_results['training_time'].apply(convert_time_to_seconds)
        sns.barplot(x='model', y='training_seconds', data=all_results, hue='model', palette='Set2', legend=False)
        plt.title('Training Time (seconds)')
        plt.xlabel('Model')
        plt.ylabel('Time (s)')
        # Hiển thị thời gian theo định dạng HH:MM:SS
        for index, row in all_results.iterrows():
            plt.text(index, row['training_seconds'], row['training_time'], 
                    ha='center', va='bottom', fontsize=10)
    else:
        sns.barplot(x='model', y=metric, data=all_results, hue='model', palette='Set2', legend=False)
        plt.title(metric)
        plt.xlabel('Model')
        plt.ylabel(metric)
        for index, row in all_results.iterrows():
            plt.text(index, row[metric], f'{row[metric]:.3f}', 
                    ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=20)

plt.suptitle('So sánh các mô hình (theo kết quả thực tế)', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('logs/compare_all_models.png', dpi=300, bbox_inches='tight')
plt.show()
print('Đã lưu biểu đồ so sánh vào logs/compare_all_models.png') 