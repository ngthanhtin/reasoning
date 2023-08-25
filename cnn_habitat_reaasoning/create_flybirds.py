# %%
# from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from PIL import Image
import requests
import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import cv2, os, json

from tqdm import tqdm
# %%
class cfg:
    dataset = 'nabirds'#inat21, cub, nabirds
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    CUB_DIR = '/home/tin/datasets/cub/CUB/test'
    NABIRD_DIR = '/home/tin/datasets/nabirds/test'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

# %%
# Load Mask2Former to detect bird
# mask2former_image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
# mask2former_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

# Load Mask2Former trained on ADE20k semantic segmentation dataset
mask2former_image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

mask2former_model.to(cfg.device)
# %% --test one image
def get_mask_one_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = mask2former_image_processor(image, return_tensors="pt").to(cfg.device)

    with torch.no_grad():
        outputs = mask2former_model(**inputs)

    # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # Perform post-processing to get semantic segmentation map
    pred_semantic_map = mask2former_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    np_img = pred_semantic_map.detach().cpu().numpy()

    return np_img

# %% get the mask object bbox
def get_mask_object_bbox(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    y, y_end = np.where(rows)[0][[0, -1]]
    x, x_end = np.where(cols)[0][[0, -1]]
    width = x_end - x + 1
    height = y_end - y + 1
    return x, y, width, height

def mask_rectangle(image, x, y, width, height):
    image[y:y+height, x:x+width] = 1
# %%-- example --
image_path = "flybird.jpeg"
image = cv2.imread(image_path)
mask = get_mask_one_image(image_path)
# %%
plt.imshow(image)
plt.axis('off')
plt.show()
# %%
print(np.unique(mask))
# mask1 = mask == 2 # sky
mask1 = np.isin(mask, [2, 90]).astype(int)
print(mask1)
print(np.unique(mask1))
plt.imshow(mask1)
plt.axis('off')
plt.show()
# %%
x, y, width, height = get_mask_object_bbox(mask1)
x,y,width, height, image.shape[:2]
# %%
# mask_rectangle(mask1, x, y, width, height)

# plt.imshow(mask1)
# plt.axis('off')
# plt.show()
# %% --save bird bounding boxes-- %%
# %%
import shutil
if cfg.dataset == 'cub':
    save_path = '/home/tin/datasets/flybird_cub_test/'
    save_path2 = '/home/tin/datasets/non_flybird_cub_test/'

    label_folders = os.listdir(cfg.CUB_DIR)

    # more: {'plant': 17, 'palm': 72, 'land': 94, 'swimming pool': 109}
    scene_dict = {'tree': 4,'sea': 26, 'river': 60, 'rock': 34, 'grass': 9, 'sand': 46, 'flower': 66, 'lake': 128, 'field': 29, 'water':21}
    scene_id = [v for k,v in scene_dict.items()]
    for folder in tqdm(label_folders):
        os.makedirs(f"{save_path}/{folder}", exist_ok=True)
        os.makedirs(f"{save_path2}/{folder}", exist_ok=True)
        image_files = os.listdir(f"{cfg.CUB_DIR}/{folder}")
        image_paths = [f"{cfg.CUB_DIR}/{folder}/{f}" for f in image_files]
        for path in image_paths:
            image = cv2.imread(path)
            img_height, img_width = image.shape[:2]
            mask = get_mask_one_image(path)
            
            sky_mask = mask == 2
            scene_mask = np.isin(mask, scene_id).astype(int)
            
            is_scene = False
            try:
                get_mask_object_bbox(scene_mask)
                is_scene = True
            except:
                is_scene = False
                print("No irrelevant scene")

            #
            if is_scene:
                shutil.copy(path, f"{save_path2}/{folder}")
                continue
            
            try:
                _, _, sky_w, sky_h = get_mask_object_bbox(sky_mask)
                # if (width*height)/(img_width*img_height) >= 0.8: # check if the image contains sky
                shutil.copy(path, f"{save_path}/{folder}")
                print('h')
            except:
                print("No sky")
                shutil.copy(path, f"{save_path2}/{folder}")
# %%
#2. NABirds bounding boxes, need to use Mask2Former to detect the mask of the bird of an image, then get the bb
import shutil
if cfg.dataset == 'nabirds':
    save_path = '/home/tin/datasets/flybird_nabirds_test/'
    save_path2 = '/home/tin/datasets/non_flybird_nabirds_test/'

    label_folders = os.listdir(cfg.NABIRD_DIR)

    # more: {'plant': 17, 'palm': 72, 'land': 94, 'swimming pool': 109}
    scene_dict = {'tree': 4,'sea': 26, 'river': 60, 'rock': 34, 'grass': 9, 'sand': 46, 'flower': 66, 'lake': 128, 'field': 29, 'water':21}
    scene_id = [v for k,v in scene_dict.items()]
    for folder in tqdm(label_folders):
        os.makedirs(f"{save_path}/{folder}", exist_ok=True)
        os.makedirs(f"{save_path2}/{folder}", exist_ok=True)
        image_files = os.listdir(f"{cfg.NABIRD_DIR}/{folder}")
        image_paths = [f"{cfg.NABIRD_DIR}/{folder}/{f}" for f in image_files]
        for path in image_paths:
            image = cv2.imread(path)
            img_height, img_width = image.shape[:2]
            mask = get_mask_one_image(path)
            
            sky_mask = mask == 2
            scene_mask = np.isin(mask, scene_id).astype(int)
            
            is_scene = False
            try:
                get_mask_object_bbox(scene_mask)
                is_scene = True
            except:
                is_scene = False
                print("No irrelevant scene")

            #
            if is_scene:
                shutil.copy(path, f"{save_path2}/{folder}")
                continue
            
            try:
                _, _, sky_w, sky_h = get_mask_object_bbox(sky_mask)
                # if (width*height)/(img_width*img_height) >= 0.8: # check if the image contains sky
                shutil.copy(path, f"{save_path}/{folder}")
                print('h')
            except:
                print("No sky")
                shutil.copy(path, f"{save_path2}/{folder}")

# %%
