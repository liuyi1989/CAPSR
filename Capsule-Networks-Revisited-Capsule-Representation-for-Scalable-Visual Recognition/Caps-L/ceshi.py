import csv
import os
# 确保目录存在
output_dir = '/home/ubuntu/lvcx/EfficientCapsNet/weights'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'training_log.csv')

# 打开文件进行写入
with open(file_path, 'w', newline='') as file:
    fieldnames = ['epoch', 'acc@1', 'acc@5', 'train_loss', 'val_loss']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # 假设这些是你从某处获取的数据
    epoch = 1
    acc1 = 0.945
    acc5 = 0.980
    train_loss = 0.05
    val_loss = 0.04

    # 写入一行数据
    writer.writerow({
        'epoch': str(epoch).zfill(3),
        'acc@1': f'{acc1:.3f}',
        'acc@5': f'{acc5:.3f}',
        'train_loss': f'{train_loss:.3f}',
        'val_loss': f'{val_loss:.3f}'
    })

# 文件自动关闭
