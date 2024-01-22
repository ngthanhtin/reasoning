# %%
import sys
sys.path.append('./Inpaint_Anything/')

import json, os, cv2, shutil
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
device = "cuda:5" if torch.cuda.is_available() else "cpu"

def inpaint_and_save(image_path, point_coords, output_dir, pre_cal_mask=None):
    point_labels=[1 for i in range(len(point_coords))]
    sam_model_type = 'vit_h'
    dilate_kernel_size = 15
    sam_ckpt = './Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth'
    lama_config = './Inpaint_Anything/lama/configs/prediction/default.yaml'
    lama_ckpt = './Inpaint_Anything/pretrained_models/big-lama'

    latest_coords = point_coords
    img = load_img_to_array(image_path)
    img_name = image_path.split('/')[-1]

    if len(img.shape) == 2: # binary
        img2 = np.zeros((img.shape[0], img.shape[1], 3))
        img2[:,:,0] = img
        img2[:,:,1] = img
        img2[:,:,2] = img
        img = img2
    if pre_cal_mask is not None:
        pre_cal_mask = np.expand_dims(pre_cal_mask, axis=0)
        pre_cal_masks = np.repeat(pre_cal_mask, 5, axis=0)
        masks = pre_cal_masks
    else:
        img_inpainted_p = f'{output_dir}/{img_name}'
        if os.path.exists(img_inpainted_p):
            return
        try:
            masks, _, _ = predict_masks_with_sam(
                img,
                latest_coords,
                point_labels,
                model_type=sam_model_type,
                ckpt_p=sam_ckpt,
                device=device,
            )
        except:
            print(latest_coords)
    masks = masks.astype(np.uint8) * 255
    
    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        if idx == 1: # only save inpaint image at index 1
            img_inpainted_p = f'{output_dir}/{img_name}'
            if os.path.exists(img_inpainted_p):
                continue
            try:
                img_inpainted = inpaint_img_with_lama(
                    img, mask, lama_config, lama_ckpt, device=device)
            except:
                print(mask.shape, img.shape, image_path)
            save_array_to_img(img_inpainted, img_inpainted_p)
# %% --inpaint CUB--
if dataset == 'cub':
    inpaint_dir = './cub_inpaint_all/'
    if not os.path.exists(inpaint_dir):
        os.makedirs(inpaint_dir)

    # get all mask file of CUB dataset
    mask_folder = '/home/tin/datasets/cub/CUB/segmentations/'
    mask_folders = os.listdir(mask_folder)
    mask_folders = [os.path.join(mask_folder, p) for p in mask_folders]
    mask_image_paths = []
    for folder in mask_folders:
        image_files = os.listdir(folder)
        image_filepaths = [os.path.join(folder, image_file) for image_file in image_files]
        mask_image_paths += image_filepaths

    # get images from retrieve folder
    image_folder_path = '../plain_clip/retrieval_cub_images_by_text/'
    image_folder_path = '/home/tin/datasets/cub/CUB/images/'

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
    kp_or_box = 'kp'

    if kp_or_box == 'kp':
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
    elif kp_or_box == 'box':
        # read nabirds bb
        nabirds_box_path = '/home/tin/datasets/nabirds/bounding_boxes.txt'
        f = open(nabirds_box_path, 'r')
        lines = f.readlines()
        image2box_dict = {}
        for i, l in enumerate(lines):
            img_name, x, y, w, h = l.split(' ')
            img_name = img_name.replace('-', '')
            x,y,w,h = int(x), int(y), int(w), int(h)
            image2box_dict[img_name] = [x,y,w,h]
        print("Finish reading bounding boxes !!!")

    # create folder to save nabirds inpainted samples
    inpaint_dir =f'nabirds_inpaint_{kp_or_box}_full/'
    if not os.path.exists(inpaint_dir):
        os.makedirs(inpaint_dir)

    # get images from retrieve folder
    # image_folder_path = '../plain_clip/retrieval_nabirds_images_by_text/'
    image_folder_path = '/home/tin/datasets/nabirds/images/'
    folders = os.listdir(image_folder_path)
    folders = [os.path.join(image_folder_path, f) for f in folders]
    
    still_left_folders = []
    inpaint_path = 'nabirds_inpaint_kp_full/'
    for orig_f in os.listdir(image_folder_path):
        if orig_f not in os.listdir(inpaint_path):
            still_left_folders.append(orig_f)
    # still_left_folders = still_left_folders[-3:]

    for i, folder in tqdm(enumerate(folders)):
        folder_name = folder.split('/')[-1]
        if folder_name not in still_left_folders:
            continue
        output_dir = inpaint_dir + '/' + folder_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = os.listdir(folder)
        for image_file in tqdm(image_files):
            if 'txt' in image_file:
                continue

            image_path = os.path.join(folder,  image_file)
            image_name = image_file.split('/')[-1]
            
            if kp_or_box == 'kp':
                kps = image2kps_dict[image_name[:-4]]
                if len(kps) == 0: # no need to inpaint
                    dest = f'{output_dir}/{img_name}'
                    shutil.copy(image_path, dest)
                else:
                    # do inpaint
                    inpaint_and_save(image_path, kps, output_dir)
            else:
                x,y,w,h = image2box_dict[image_name[:-4]]
                img = cv2.imread(image_path)
                h,w = img.shape[:2]
                mask = np.zeros((h, w))
                mask[y:y+h, x:x+w] = 1
                # do inpaint
                inpaint_and_save(image_path, [0,0], output_dir, pre_cal_mask=mask)

# --inpaint inat--
elif dataset == 'inat21':
    inpaint_dir = './inat21_inpaint_all/'
    if not os.path.exists(inpaint_dir):
        os.makedirs(inpaint_dir)

    # get all mask file of INaturalist 2021 Onlybird dataset
    mask_folder = '/home/tin/datasets/inaturalist2021_onlybird/inat_masks/'
    mask_folders = os.listdir(mask_folder)
    mask_folders = [os.path.join(mask_folder, p) for p in mask_folders]
    mask_image_paths = []
    for folder in mask_folders:
        image_files = os.listdir(folder)
        image_filepaths = [os.path.join(folder, image_file) for image_file in image_files]
        mask_image_paths += image_filepaths
    # create a dict for quick look-up
    mask_name_path_dict = {k.split('/')[-1][:-4]:k for k in mask_image_paths}
    
    # get images from retrieve folder
    # image_folder_path = '../plain_clip/retrieval_cub_images_by_text/'
    image_folder_path = '/home/tin/datasets/inaturalist2021_onlybird/bird_train/' # all

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
            mask_path = mask_name_path_dict[image_name[:-4]]
            mask = cv2.imread(mask_path, 0)

            image_path = os.path.join(folder,  image_file)
            # plt.imshow(mask)
            # plt.axis('off')
            # plt.show()
            
            # do inpaint
            inpaint_and_save(image_path, [0,0], output_dir, pre_cal_mask=mask)
# %%
