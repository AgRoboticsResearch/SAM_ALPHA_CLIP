import torch
import alpha_clip
from PIL import Image
import numpy as np
import random
from torchvision import transforms
import cv2
import os
import datetime

np.set_printoptions(suppress=True)

from image_utils import mask_image, crop_object_from_white_background

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = alpha_clip.load(
    "ViT-L/14", alpha_vision_ckpt_pth="../checkpoints/clip_l14_grit20m_fultune_2xe.pth",
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
    if label == 'strawberry':
            texts_str = ['a red strawberry', 'a green strawberry', 'shadow']
            #texts_str = ['a strawberry predominantly red in color', 'a strawberry predominantly green in color']
            text_str = ['ripe', 'unripe', 'shadow']
            text_tokens_new = alpha_clip.tokenize([desc for desc in texts_str]).to(device)
            with torch.no_grad():
                image_features_new = model.visual(image, alpha)
                text_features_new = model.encode_text(text_tokens_new)
            image_features_new /= image_features_new.norm(dim=-1, keepdim=True)
            text_features_new /= text_features_new.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features_new @ text_features_new.T).softmax(dim=-1).cpu().numpy()
            label_new = text_str[np.argmax(similarity)]
            return label_new
    else:
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

image_segs_folder = "F:\doctor\strawberry\segment-anything-main\segment-anything-main\img2"
texts = [
    "a strawberry with numerous tiny seeds",
    "a green leaf or leaves with jagged edges",
    "a green long and thin stem",
    "a white flower",
    "a yellow stone or black plastic or soil or something else",
]
text = ['strawberry', 'leaf','stem','flower','others']
t1 = datetime.datetime.now()
now_folder = t1.strftime('%Y%m%d%H%M')
output_path = os.path.join('F:\doctor\distill_fundation_model\output', now_folder)
os.makedirs(output_path, exist_ok= False)
#print(output_path)
files_img = os.listdir(image_segs_folder)
for file_img in files_img:
    image_path = os.path.join(image_segs_folder, file_img)
    id_new = str(file_img)[:-4]
    mask_seg_folder = f"F:\doctor\strawberry\segment-anything-main\segment-anything-main\output/202403191204/{id_new}/"
    files = os.listdir(mask_seg_folder)
    # replace this with your desired output path
    image = Image.open(image_path).convert('RGB')
    results = []
    for file in files:
        if file == "mask_0.png" or file == f"{id_new}_all_masks.png":
            continue

        mask_path = mask_seg_folder + file
        
        mask = np.array(Image.open(mask_path)) 

        cropped_img, cropped_mask, ymin, xmin, ymax, xmax = crop_object_from_mask(image, mask)
        label = alpha_clip_prediction(cropped_img, cropped_mask, texts, text)
        print(label)
        results.append({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    img_final = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    for res in results:
        #if res['label']=='ripe' or res['label']=='unripe' :
        cv2.rectangle(img_final, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), (0, 0, 255), 2)  # Red rectangles

        # Add label with white background
        (label_width, label_height), baseline = cv2.getTextSize(res['label'], font, font_scale, thickness)
        cv2.rectangle(img_final, (res['xmin'], res['ymin']-label_height-baseline), (res['xmin']+label_width, res['ymin']), (255,255,255), cv2.FILLED)
        cv2.putText(img_final, res['label'], (res['xmin'], res['ymin']-10), font, font_scale, (0,0,255), thickness)  # Red text
    cv2.imwrite(os.path.join(output_path, f"{id_new}_label.png"), img_final)
    '''
    for res in results:
        #cv2.rectangle(img_final, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), (0, 0, 255), 2)
        #cv2.putText(img_final, res['label'], (res['xmin'], res['ymin']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(img_final, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), (0, 0, 255), 2)  # Red rectangles

        # Add label with white background
        (label_width, label_height), baseline = cv2.getTextSize(res['label'], font, font_scale, thickness)
        cv2.rectangle(img_final, (res['xmin'], res['ymin']-label_height-baseline), (res['xmin']+label_width, res['ymin']-10), (255,255,255), cv2.FILLED)
        cv2.putText(img_final, res['label'], (res['xmin'], res['ymin']-10), font, font_scale, (0,0,255), thickness)  # Red text
    cv2.imwrite(os.path.join(output_path, f"{id_new}_label.png"), img_final)
print(datetime.datetime.now(), 'Done')'''
