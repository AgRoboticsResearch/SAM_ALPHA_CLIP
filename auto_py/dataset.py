import os
import shutil
import numpy as np

def split_dataset_into_train_val_test(dataset_dir, output_dir, train_ratio, val_ratio, test_ratio):

    # 定义对应的文件夹名
    classes_dir = os.listdir(dataset_dir)

    # 创建train, val, test文件夹
    for set_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, set_name), exist_ok=True)
    
    # 初始化随机数种子
    np.random.seed(0)
    
    for cls in classes_dir:
        
        # 获取类别文件夹下所有图像
        img_files = os.listdir(os.path.join(dataset_dir, cls))
        
        # 随机洗牌图像列表
        np.random.shuffle(img_files)

        # 计算训练、验证和测试集的大小
        train_size = int(len(img_files) * train_ratio)
        val_size = int(len(img_files) * val_ratio)
        
        for i, img_file in enumerate(img_files):
            # 根据索引划分图像
            if i < train_size:
                dst_dataset_dir = os.path.join(output_dir, 'train', cls)
            elif i < (train_size + val_size):
                dst_dataset_dir = os.path.join(output_dir, 'val', cls)
            else:
                dst_dataset_dir = os.path.join(output_dir, 'test', cls)

            os.makedirs(dst_dataset_dir, exist_ok=True)
            # 将图像拷贝到对应的文件夹
            shutil.copy(os.path.join(dataset_dir, cls, img_file), os.path.join(dst_dataset_dir, img_file))

dataset_dir = 'path_to_your_dataset_dir'  
output_dir = '../output_img'
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
split_dataset_into_train_val_test(dataset_dir, output_dir, train_ratio, val_ratio, test_ratio)