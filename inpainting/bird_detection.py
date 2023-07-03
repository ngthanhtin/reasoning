# %%
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

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
    device = "cuda:4" if torch.cuda.is_available() else "cpu"

    CUB_DIR = '/home/tin/datasets/CUB_200_2011/'
    NABIRD_DIR = '/home/tin/datasets/nabirds/'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

# %%
# Load Mask2Former to detect bird
mask2former_image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
mask2former_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
mask2former_model.to(cfg.device)
# %% --test one image
def get_mask_one_image(image_path):
    image = Image.open(image_path)
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
image_path = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/04596_Animalia_Chordata_Aves_Trogoniformes_Trogonidae_Trogon_rufus/fc2c1c7b-693e-417c-8130-4e6fd5b36d72.jpg"
# image_path = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/04596_Animalia_Chordata_Aves_Trogoniformes_Trogonidae_Trogon_rufus/ef5ca918-bfcd-4156-888c-d94a7a0f5a9a.jpg"
# image_path = 'test_bird.jpeg'
image = cv2.imread(image_path)
mask = get_mask_one_image(image_path)
# %%
plt.imshow(image)
plt.axis('off')
plt.show()
# %%
mask1 = mask == 14 # bird (coco)
# mask2 = mask == 116
# mask3 = mask == 125

print(np.unique(mask1))
plt.imshow(mask1)
plt.axis('off')
plt.show()
# %%
x, y, width, height = get_mask_object_bbox(mask1)
# %%
mask_rectangle(mask1, x, y, width, height)

plt.imshow(mask1)
plt.axis('off')
plt.show()
# %% --save bird bounding boxes-- %%
# %%
#1. CUB bounding boxes, for CUB, I already have the mask images, just take it and get the bounding box
# get the image path, extract bb, save to a json file
if cfg.dataset == 'cub':
    imagedir_bb_dict = {}
    cub_mask_dir = '/home/tin/datasets/CUB_200_2011/segmentations/'
    folders = os.listdir(cub_mask_dir)
    folders = [os.path.join(cub_mask_dir, f) for f in folders]

    for folder_path in folders:
        image_paths = os.listdir(folder_path)
        image_paths = [os.path.join(folder_path, f) for f in image_paths]
        for path in tqdm(image_paths):
            mask = cv2.imread(path, 0)
            x,y,w,h = get_mask_object_bbox(mask)
            imagedir_bb_dict[path] = [int(x),int(y),int(w),int(h)]
    
    with open(f'{cfg.dataset}_bird_bb.json', "w") as json_file:
        json.dump(imagedir_bb_dict, json_file, indent=4)

# %%
#2. NABirds bounding boxes, need to use Mask2Former to detect the mask of the bird of an image, then get the bb
if cfg.dataset == 'nabirds':
    imagedir_bb_dict = {}

    nabird_image_dir = cfg.NABIRD_DIR + '/images/'
    folders = os.listdir(nabird_image_dir)
    folders = [os.path.join(nabird_image_dir, f) for f in folders]

    for folder_path in tqdm(folders):
        image_paths = os.listdir(folder_path)
        image_paths = [os.path.join(folder_path, f) for f in image_paths]
        for path in image_paths:
            image = cv2.imread(image_path)
            mask = get_mask_one_image(image_path)
            mask = mask == 14 # get the mask of the bird
            x, y, w, h = get_mask_object_bbox(mask)
            mask_rectangle(mask1, x, y, w, h)
            
            imagedir_bb_dict[path] = [int(x),int(y),int(w),int(h)]

    with open(f'{cfg.dataset}_bird_bb.json', "w") as json_file:
        json.dump(imagedir_bb_dict, json_file, indent=4)
# %%
#3. INat21 bounding boxes, need to use Mask2Former to detect the mask of the bird of an image, then get the bb
if cfg.dataset == 'inat21':
    imagedir_bb_dict = {}

    inat_image_dir = cfg.INATURALIST_DIR + '/bird_train/'
    folders = os.listdir(inat_image_dir)
    folders = [os.path.join(inat_image_dir, f) for f in folders]

    for folder_path in tqdm(folders):
        image_paths = os.listdir(folder_path)
        image_paths = [os.path.join(folder_path, f) for f in image_paths]
        for path in image_paths:
            image = cv2.imread(image_path)
            mask = get_mask_one_image(image_path)
            mask = mask == 14 # get the mask of the bird
            x, y, w, h = get_mask_object_bbox(mask)
            mask_rectangle(mask1, x, y, w, h)
            
            imagedir_bb_dict[path] = [int(x),int(y),int(w),int(h)]

    with open(f'{cfg.dataset}_bird_bb.json', "w") as json_file:
        json.dump(imagedir_bb_dict, json_file, indent=4)
# %%
