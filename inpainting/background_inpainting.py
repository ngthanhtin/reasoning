# %%
import sys
sys.path.append('./Inpaint_Anything/')

import json, os, cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from Inpaint_Anything.sam_segment import predict_masks_with_sam
from Inpaint_Anything.lama_inpaint import inpaint_img_with_lama
from Inpaint_Anything.utils import load_img_to_array, save_array_to_img, dilate_mask

# %% init inpainting module
dataset = 'cub'#inat21, cub, nabirds
device = "cuda:6" if torch.cuda.is_available() else "cpu"

def inpaint_and_save(image_path, point_coords, output_dir):
    point_labels=[1]
    sam_model_type = 'vit_h'
    dilate_kernel_size = 15
    sam_ckpt = './Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth'
    lama_config = './Inpaint_Anything/lama/configs/prediction/default.yaml'
    lama_ckpt = './Inpaint_Anything/pretrained_models/big-lama'

    latest_coords = point_coords
    img = load_img_to_array(image_path)

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    img_name = image_path.split('/')[-1]

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        if idx == 1: # only save inpaint image at index 1
            img_inpainted_p = output_dir + '/' + img_name
            img_inpainted = inpaint_img_with_lama(
                img, mask, lama_config, lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)
# %% --inpaint CUB--
inpaint_dir = './cub_inpaint/'
if not os.path.exists(inpaint_dir):
    os.makedirs(inpaint_dir)

with open('cub_bird_bb.json', 'r') as f:
    cub_bird_bb = json.load(f)
# remove relative paths
cub_bird_bb = {k.split("/")[-1]:v for k, v in cub_bird_bb.items()}
image_folder_path = '../plain_clip/retrieval_cub_images_by_text/'
folders = os.listdir(image_folder_path)
folders = [os.path.join(image_folder_path, f) for f in folders]
for i, folder in tqdm(enumerate(folders)):
    folder_name = folder.split('/')[-1]
    output_dir = inpaint_dir + '/' + folder_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(folder)
    for image_file in image_files:
        if 'txt' in image_file:
            continue
        try:
            x,y,w,h = cub_bird_bb[image_file]
        except:
            x,y,w,h = cub_bird_bb[image_file[:-4] + '.png']
        image_path = os.path.join(folder,  image_file)

        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        
        # do inpaint
        inpaint_and_save(image_path, [x+w//2, y+h//2], output_dir)
# %% --inpaint nabirds--
with open('nabirds_bird_bb.json', 'r') as f:
    nabirds_bird_bb = json.load(f)

# %% --inpaint inat--
with open('inat21_bird_bb.json', 'r') as f:
    inat_bird_bb = json.load(f)