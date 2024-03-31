import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('F:\doctor\StrawDI_Data\images/train/2650.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, _ = img.shape

txt_pred_path = 'F:\doctor\distill_fundation_model\output_1_1/train/2650.txt'
txt_true_path = 'F:\doctor\StrawDI_Data\labels/train/1.txt'
with open(txt_pred_path, 'r') as file:
    for line in file:
        # 分割行以获取类别和坐标
        data = line.split()
        category = int(data[0])
        coordinates = data[1:]

        # 确保坐标数量是偶数（x, y对）
        assert len(coordinates) % 2 == 0

        # 取两两一组作为坐标
        for i in range(0, len(coordinates), 2):
            x = int(float(coordinates[i]) * width) # 横坐标 x 转化为实际像素，乘以图片宽度
            y = int(float(coordinates[i + 1]) * height) # 纵坐标 y 转化为实际像素，乘以图片高度
            
            # 在图像上标记坐标
            img = cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
            
            # 添加类别标签
        cv2.putText(img, str(category), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# 显示图像
plt.imshow(img)
plt.show()
