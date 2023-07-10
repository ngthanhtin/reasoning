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
dataset = 'nabirds' # inat21, cub, nabirds
device = "cuda:6" if torch.cuda.is_available() else "cpu"

def inpaint_and_save(image_path, point_coords, output_dir, pre_cal_mask=None):
    point_labels=[1 for i in range(len(point_coords))]
    sam_model_type = 'vit_h'
    dilate_kernel_size = 15
    sam_ckpt = './Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth'
    lama_config = './Inpaint_Anything/lama/configs/prediction/default.yaml'
    lama_ckpt = './Inpaint_Anything/pretrained_models/big-lama'

    latest_coords = point_coords
    img = load_img_to_array(image_path)
    if pre_cal_mask is not None:
        pre_cal_mask = np.expand_dims(pre_cal_mask, axis=0)
        pre_cal_masks = np.repeat(pre_cal_mask, 5, axis=0)
        masks = pre_cal_masks
    else:
        masks, _, _ = predict_masks_with_sam(
            img,
            latest_coords,
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
if dataset == 'cub':
    inpaint_dir = './cub_inpaint_new/'
    if not os.path.exists(inpaint_dir):
        os.makedirs(inpaint_dir)

    # get all mask file of CUB dataset
    mask_folder = '/home/tin/datasets/CUB_200_2011/segmentations/'
    mask_folders = os.listdir(mask_folder)
    mask_folders = [os.path.join(mask_folder, p) for p in mask_folders]
    mask_image_paths = []
    for folder in mask_folders:
        image_files = os.listdir(folder)
        image_filepaths = [os.path.join(folder, image_file) for image_file in image_files]
        mask_image_paths += image_filepaths

    # get images from retrieve folder
    image_folder_path = '../plain_clip/retrieval_cub_images_by_text/'
    folders = os.listdir(image_folder_path)
    folders = [os.path.join(image_folder_path, f) for f in folders]

    for i, folder in tqdm(enumerate(folders)):
        folder_name = folder.split('/')[-1]
        output_dir = inpaint_dir + '/' + folder_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = os.listdir(folder)
        for image_file in tqdm(image_files):
            if 'txt' in image_file:
                continue

            image_name = image_file.split('/')[-1]
            mask = None
            for mask_path in mask_image_paths:
                if image_name[:-4] in mask_path:
                    mask = cv2.imread(mask_path, 0)
                    break

            image_path = os.path.join(folder,  image_file)
            # plt.imshow(mask)
            # plt.axis('off')
            # plt.show()
            
            # do inpaint
            inpaint_and_save(image_path, [0,0], output_dir, pre_cal_mask=mask)
#  --inpaint nabirds--
elif dataset == 'nabirds':
    # read nabirds keypoints
    f = open('/home/tin/datasets/nabirds/parts/part_locs.txt', 'r')
    lines = f.readlines()
    image2kps_dict = {}
    for i, l in enumerate(lines):
        img_name, _, x, y, visible = l.split(' ')
        img_name = img_name.replace('-', '')
        if img_name not in image2kps_dict:
            image2kps_dict[img_name] = []
        if int(visible) == 1:
            image2kps_dict[img_name].append([float(x),float(y)])
    print("Finish reading keypoints !!!")
    # create folder to save nabirds inpainted samples
    inpaint_dir ='nabirds_inpaint/'
    if not os.path.exists(inpaint_dir):
        os.makedirs(inpaint_dir)

    # get images from retrieve folder
    image_folder_path = '../plain_clip/retrieval_nabirds_images_by_text/'
    folders = os.listdir(image_folder_path)
    folders = [os.path.join(image_folder_path, f) for f in folders]

    for i, folder in tqdm(enumerate(folders)):
        folder_name = folder.split('/')[-1]
        output_dir = inpaint_dir + '/' + folder_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = os.listdir(folder)
        for image_file in tqdm(image_files):
            if 'txt' in image_file:
                continue

            image_name = image_file.split('/')[-1]
            
            kps = image2kps_dict[image_name[:-4]]
            image_path = os.path.join(folder,  image_file)
            
            # do inpaint
            inpaint_and_save(image_path, kps, output_dir)

# --inpaint inat--
elif dataset == 'inat21':
    with open('inat21_bird_bb.json', 'r') as f:
        inat_bird_bb = json.load(f)
# %%
