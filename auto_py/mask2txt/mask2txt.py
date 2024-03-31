import os
import cv2
import numpy as np
import datetime
from torchvision import transforms
import torch
import alpha_clip
from PIL import Image
from scipy.ndimage import label as label_region

label_dict = {"ripe": 0, "unripe": 1, "leaf": 2, "stem": 3, "flower": 4,"others": 5}
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = alpha_clip.load(
    "ViT-L/14", alpha_vision_ckpt_pth="F:\doctor\distill_fundation_model\checkpoints\clip_l14_grit20m_fultune_2xe.pth",
    device=device
)

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

def alpha_clip_prediction(image, mask, texts, text):
    text_tokens = alpha_clip.tokenize([desc for desc in texts]).to(device)
    image = preprocess(image.copy()).unsqueeze(0).half().to(device)

    if len(mask.shape) == 2: 
        binary_mask = (mask == 255)
    if len(mask.shape) == 3: 
        binary_mask = (mask[:, :, 0] == 255)
    
    alpha = mask_transform((binary_mask * 255).astype(np.uint8))
    alpha = alpha.half().cuda().unsqueeze(dim=0)

    with torch.no_grad():
        image_features = model.visual(image, alpha)
        text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    label = text[np.argmax(similarity)]
    return label

def crop_object_from_mask(image, mask):
    valid_mask = np.any(mask != 0, axis=2)
    ymin, xmin = np.where(valid_mask)[0].min(), np.where(valid_mask)[1].min()
    ymax, xmax = np.where(valid_mask)[0].max() + 1, np.where(valid_mask)[1].max() + 1
    cropped_mask = Image.fromarray(mask)
    cropped_img = image.crop((xmin, ymin, xmax, ymax))
    cropped_mask = cropped_mask.crop((xmin, ymin, xmax, ymax))
    cropped_mask = np.asarray(cropped_mask)
    return cropped_img, cropped_mask, ymin, xmin, ymax, xmax

def pixel_num(mask):
    mask[mask == 255] = 1
    white_pixels = np.sum(mask == 1)
    num = int(white_pixels/5)
    return num



texts = [
    "a red strawberry",
    "a green strawberry",
    "a green leaf or leaves with jagged edges",
    "a green long and thin stem",
    "a white flower",
    "a yellow stone or black plastic or soil or something else",
]
text = ['ripe', 'unripe', 'leaf','stem','flower','others']

image_segs_folder = "F:\doctor\strawberry\StrawDI_Db1\StrawDI_Db1"
output_path = 'F:\doctor\distill_fundation_model\output_1_1'

files_dir = os.listdir(image_segs_folder)  #train val test

for file_dir in files_dir:
    #print(os.path.join(image_segs_folder, file_dir, 'img'))
    output_txt_path = os.path.join(output_path, file_dir)
    os.makedirs(output_txt_path, exist_ok=True)
    print(output_txt_path)

    file_train = os.listdir(os.path.join(image_segs_folder, file_dir, 'img')) #StrawDI_Db1/train
    for file_img in file_train:
        
        image_path = os.path.join(image_segs_folder, file_dir, 'img', file_img)  #StrawDI_Db1/train/img/1.png
        stem, suffix = os.path.splitext(file_img)
        mask_seg_folder = f"F:\doctor\distill_fundation_model\output_all\output_mask/{file_dir}/{stem}/"
        files = os.listdir(mask_seg_folder)
        # open the image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        results = []
        file_contents = []  # dynamically save results before writing to file

        for file in files:
            
            # read the mask and find all pixels that equal 1
            if file == 'mask_0,png':
                continue
            else:
                mask_path = mask_seg_folder + file
                #flag = eliminate_disconnected(mask_path)
                mask = cv2.imread(mask_path, 0)
            
                #保留最大联通区域
                labelled_mask, num_labels = label_region(mask)
                region_sizes = np.bincount(labelled_mask.flat)
                # the first region is the background, so we ignore it by setting its size to 0
                region_sizes[0] = 0
                # find the label of the largest region
                max_region_label = np.argmax(region_sizes)

                # create a new mask where only the largest region is white
                mask = ((labelled_mask == max_region_label) * 255).astype(np.uint8)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                c = max(contours, key = cv2.contourArea)
                c = c.reshape(-1, 2)  # 转变形状以便处理

                # 在所有点中以一定间隔取点
                num_points = len(c)
                skip = num_points // 300
                skip = max(1, skip)  # 确保步长至少为1
                approx_sparse = c[::skip]

                # 找到最低点（也就是y坐标最大的点，并假设图像中只有一个）
                bottom_point_index = np.argmax(approx_sparse[:, 1])

                sorted_points = np.concatenate([approx_sparse[bottom_point_index:], approx_sparse[:bottom_point_index]])
                # normalize the coordinates.
            

                # get the label and process the image
                mask_pro = np.array(Image.open(mask_path))
                cropped_img, cropped_mask, ymin, xmin, ymax, xmax = crop_object_from_mask(image, mask_pro)
                label = alpha_clip_prediction(cropped_img, cropped_mask, texts, text)
                label_num = label_dict[label]
                

                # prepare the line for the file
                line = f'{label_num} ' + ' '.join(f'{point[0]/img_width} {point[1]/img_height}' for point in sorted_points) + '\n'
                file_contents.append(line)


            # write all lines to file at once
            filename = os.path.join(output_txt_path, f'{stem}.txt')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.writelines(file_contents)


print(datetime.datetime.now(), 'Done')