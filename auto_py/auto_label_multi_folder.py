from IPython.display import display, HTML

from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from pathlib import Path
import sys
import numpy as np
import datetime
from torchvision import transforms
import torch
import alpha_clip
from PIL import Image
from scipy.ndimage import label as label_region



from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def save_mask(anns, image, path, basename):

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(sorted_anns):
        #a = ann['original_index']
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)   #如果不进行remove处理，这句不用注释

        img = (mask*255).astype(np.uint8)  # Setting mask as white
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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


now = datetime.datetime.now()

sam_checkpoint = "F:\doctor\distill_fundation_model\checkpoints\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

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



texts = [
    "a red strawberry",
    "a green strawberry",
    "a green leaf or leaves with jagged edges",
    "a green long and thin stem",
    "a white flower",
    "a yellow stone or black plastic or soil or something else",
]
text = ['ripe', 'unripe', 'leaf','stem','flower','others']

imgdir_path = 'F:\doctor\strawberry\StrawDI_Db1\StrawDI_Db1'   #修改成你自己的原始图像图像路径
img_dir = os.listdir(imgdir_path)  #train val text
out_folder = 'F:\doctor\distill_fundation_model\output_all\output_mask'  #修改成你保存分割掩码的路径
for file_train in img_dir:
    mask_folder_path = os.path.join(out_folder, file_train)
    print(mask_folder_path) 
    os.makedirs(mask_folder_path, exist_ok=True)
    files_img = os.path.join(imgdir_path, file_train, 'img')
    files = os.listdir(files_img) 
    for file in files:
        image_path = os.path.join(files_img, file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        basename = Path(image_path).stem  #提取无后缀的文件名
        os.makedirs(f'{mask_folder_path}/{basename}', exist_ok=True)
        path_stem = f'{mask_folder_path}/{basename}'
        # Create folder for masks

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.95,
        stability_score_offset = 1.0,
        box_nms_thresh = 0.2,
        crop_n_layers=0,
        crop_nms_thresh = 0.7,
        crop_overlap_ratio = 500 / 1500,
        crop_n_points_downscale_factor=0,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
        args = mask_generator_2.__dict__

        # Construct the content string by joining the arguments and their values


        masks2 = mask_generator_2.generate(image)
        #masks2 = filter_small_masks(masks2_1, 30*30)

        save_mask(masks2, image, path_stem, basename)
    str_file = str(file)[:-4]
    label_output_path = os.path.join('F:\doctor\distill_fundation_model\output_all\output_label', file_train)  #修改成你自己的保存txt文件的路径
    #print('label_output_path', label_output_path)
    os.makedirs(label_output_path, exist_ok=True)
    #print('line 160 mask_folder_path', mask_folder_path)
    files_mask = os.listdir(mask_folder_path)
    #print('files_mask',files_mask)

    for file_mask in files_mask:
        #print('file_mask',file_mask)
        #进入mask文件夹，由命名为1 2 3...n的文件夹组成，每个文件夹里面存放对应n.png的mask_i
        image_path1 = os.path.join(imgdir_path, file_train, 'img', file_mask + '.png')
        #print('image_path1',image_path1)
        image = Image.open(image_path1).convert('RGB')
        img_width, img_height = image.size
        file_mask_dir = os.path.join(mask_folder_path, file_mask)
        #print('file_mask_dir',file_mask_dir)
        files_mask2 = os.listdir(file_mask_dir)
        results = []
        file_contents = []
        for file_mask2 in files_mask2:
            if file_mask2 == 'mask_0.png':
                continue
            else:
                mask_img = os.path.join(file_mask_dir ,file_mask2)
                mask = cv2.imread(mask_img, 0)

                #保留最大联通区域
                labelled_mask, num_labels = label_region(mask)
                region_sizes = np.bincount(labelled_mask.flat)
                # the first region is the background, so we ignore it by setting its size to 0
                region_sizes[0] = 0
                # find the label of the largest region
                max_region_label = np.argmax(region_sizes)

                # create a new mask where only the largest region is white
                mask = ((labelled_mask == max_region_label) * 255).astype(np.uint8)

                #print(mask.size)
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
                mask_pro = np.array(Image.open(mask_img))
                cropped_img, cropped_mask, ymin, xmin, ymax, xmax = crop_object_from_mask(image, mask_pro)
                label = alpha_clip_prediction(cropped_img, cropped_mask, texts, text)
                label_num = label_dict[label]
                

                # prepare the line for the file
                line = f'{label_num} ' + ' '.join(f'{format(point[0]/img_width,".6f")} {format(point[1]/img_height, ".6f")}' for point in sorted_points) + '\n'
                file_contents.append(line)

            # write all lines to file at once
            filename = os.path.join(label_output_path, f'{file_mask}.txt')
            #print('txt的filename', filename)
            with open(filename, 'w') as f:
                f.writelines(file_contents)
            #print(file_mask,'have been finished!')

print(datetime.datetime.now(), 'Done!')
        



