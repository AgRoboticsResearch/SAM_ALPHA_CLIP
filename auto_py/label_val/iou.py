import cv2
import numpy as np
import os

def mask_valise(txt_path, category_target):
    mask_total = np.zeros((height, width), dtype=np.uint8)
    
    with open(txt_path, 'r') as file:
        for line in file:
            # 分割行以获取类别和坐标
            data = line.split()
            category = int(data[0])
            coordinates = data[1:]

            # 确保坐标数量是偶数（x, y对）
            assert len(coordinates) % 2 == 0

            contour = []
            # 取两两一组作为坐标
            for i in range(0, len(coordinates), 2):
                x = int(float(coordinates[i]) * width)
                y = int(float(coordinates[i + 1]) * height)
                contour.append([x, y])

            # 转换为numpy array
            contour = np.array(contour)
            # 初始化掩膜
            mask = np.zeros((height, width), dtype=np.uint8)
            if category == category_target:
                cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)
                mask_total[mask==255] = 1

    return mask_total

def calculate_iou(mask_true, mask_pred):
    intersect = np.sum(mask_true + mask_pred == 2)
    union = np.sum(mask_true + mask_pred > 0)
    if union == 0:
        iou = 1
    else:
        iou = intersect / union
    return iou

# 读取图像
txt_true_folder = 'F:\\doctor\\StrawDI_Data\\labels\\train'
txt_pred_folder = 'F:\\doctor\\distill_fundation_model\\data\\labels\\train'
img_folder = 'F:\\doctor\\distill_fundation_model\\data\\images\\train'
img_files = os.listdir(img_folder)
iou_list_category_0 = []
iou_list_category_1 = []
for file in img_files:
    img = cv2.imread(os.path.join(img_folder, file))
    height, width, _ = img.shape
    stem, suffix = os.path.splitext(file)

    txt_true_path = os.path.join(txt_true_folder,f'{stem}.txt')
    txt_pred_path = os.path.join(txt_pred_folder,f'{stem}.txt')

    for category_target in [0, 1]:
        # Generate mask for true image
        mask_true = mask_valise(txt_true_path, category_target)
        # Generate mask for predicted image
        mask_pred = mask_valise(txt_pred_path, category_target)

        iou = calculate_iou(mask_true, mask_pred)
        if category_target == 0:
            iou_list_category_0.append(iou)
        else:
            iou_list_category_1.append(iou)

iou_0 = sum(iou_list_category_0) / len(iou_list_category_0)
iou_1 = sum(iou_list_category_1) / len(iou_list_category_1)
print('The average IoU for Class 0 is: ', iou_0)
print('The average IoU for Class 1 is: ', iou_1)