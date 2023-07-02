#%%
import os, sys, json, cv2
import natsort
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import random

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from tqdm import tqdm

import clip

from datasets import CUBDataset, NABirdsDataset, INaturalistDataset
# %%
def seed_everything(seed: int):
    # import random, os
    # import numpy as np
    # import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(128)

class cfg:
    dataset = 'cub'
    batch_size = 64
    device = "cuda:4" if torch.cuda.is_available() else "cpu"

    CUB_DIR = '/home/tin/datasets/CUB_200_2011/'
    NABIRD_DIR = '/home/tin/datasets/nabirds/'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

    MODEL_TYPE = 'ViT-B/32'
    IMAGE_SIZE = 224

# %%
# init CLIP
model, preprocess = clip.load(cfg.MODEL_TYPE, device=cfg.device, jit=False)
# %%
# create dataset and dataloder

if cfg.dataset == 'cub':
    # load CUB dataset
    dataset_dir = pathlib.Path(cfg.CUB_DIR)
    dataset = CUBDataset(dataset_dir, train=False, transform=preprocess)

# elif cfg.dataset == 'nabirds':
#     dataset_dir = pathlib.Path(cfg.NABIRD_DIR)
#     dataset = NABirdsDataset(dataset_dir, train=False, subset_class_names=subset_class_names, transform=preprocess)

# elif cfg.dataset == 'inaturalist2021':
#     dataset_dir = pathlib.Path(cfg.INATURALIST_DIR)
#     dataset = INaturalistDataset(root_dir=dataset_dir, train=False, subset_class_names=subset_class_names, n_pixel=hparams['image_size'], transform=preprocess)


dataloader = DataLoader(dataset, cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True)

# %%
def compute_text_feature(desc, cut_len = 250):
    
    if len(desc) >= cut_len:
        desc = desc[:cut_len]

    tokens = clip.tokenize(desc).to(cfg.device)
    return F.normalize(model.encode_text(tokens)).detach().cpu().numpy(), desc

def compute_image_feature(image):
    """input: array: (W, H, C)"""
    image = preprocess(image)
    image = image.unsqueeze(0).to(cfg.device)
    image_feat = model.encode_image(image).to(cfg.device)
    return F.normalize(image_feat).detach().cpu().numpy()

def compute_image_features(loader):
    """
    compute image features and return them with their respective image paths
    """
    image_features = []
    paths = []
    for i, batch in enumerate(tqdm(loader)):
        images, _, _paths = batch
        paths += _paths
        images = images.to(cfg.device)
        features = model.encode_image(images)
        features = F.normalize(features)
        image_features.extend(features.detach().cpu().numpy())
    return np.array(image_features), paths

# %%
# image retrieval based on image-text
def find_image_by_text(text_query, image_features, image_paths, n=1):
    zeroshot_weights, text_query_after = compute_text_feature(text_query)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append(image_paths[idx])
    return file_paths, text_query_after

# %%
# image retrieval based on image-image
def find_image_by_image(image_path, image_features, image_paths, n=1):
    image = Image.open(image_path)
    zeroshot_weights = compute_image_feature(image)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append(image_paths[idx])
    return file_paths
# %%
from PIL import Image
def show_images(image_list):
    for im_path in image_list:
        print(im_path)
        image = Image.open(im_path)
        plt.imshow(image)
        plt.show()
# %%
image_features, image_paths = compute_image_features(dataloader)

# %% test retrieving image by text
text = "this bird live in open areas with thick, low vegetation, ranging from marsh to grassland to open pine forest. During migration, they use an even broader suite of habitats including backyards and forest"
returned_image_paths, text_after = find_image_by_text(text, image_features, image_paths, n=5)
print(f"Before: {text}")
print(f"After: {text_after}")

# %%
show_images(returned_image_paths)

# %% test retrieving image by image
image_path = 'test_bird.jpeg'
returned_image_paths = find_image_by_image(image_path, image_features, image_paths, n=5)
# %%
show_images(returned_image_paths)
# %% --- get the habitat description of 200 cub classes ---
f = open("./descriptors/cub/additional_chatgpt_descriptors_cub.json", 'r')
data = json.load(f)
data = {k: v[-1][9:] for k,v in data.items()}
data
# %% each class retrieves 5 images
import shutil, os
save_retrieved_path = "retrieval_cub_images_by_text/"    
if not os.path.exists(save_retrieved_path):
    os.makedirs(save_retrieved_path)

retrieval_acc_dict = {}
for k, v in data.items():
    # v = v.replace(k, 'this bird')
    class_name = k.replace('-', ' ').lower()

    if class_name not in retrieval_acc_dict:
        retrieval_acc_dict[class_name] = 0

    if not os.path.exists(os.path.join(save_retrieved_path, k)):
        os.makedirs(os.path.join(save_retrieved_path, k))
    returned_image_paths, v_after = find_image_by_text(v, image_features, image_paths, n=5)
    # save image and query
    for p in returned_image_paths:
        shutil.copy(p, os.path.join(save_retrieved_path, k))
        retrieved_image_class_name = p.split('/')[-1].split('_')[:-2]
        retrieved_image_class_name = " ".join(retrieved_image_class_name).lower()

        if retrieved_image_class_name == class_name:
            retrieval_acc_dict[class_name] += 1
    with open(f'{os.path.join(save_retrieved_path, k)}/query.txt', 'w') as f:
        f.write(v)
        f.write('\n')
        f.write(v_after)

retrieval_acc_dict = {k: v/5 for k, v in retrieval_acc_dict.items()}

retrieval_acc_dict  

# %% statistic
avg_acc = 0
classes_1 = []
classes_0 = []
for k, v in retrieval_acc_dict.items():
    avg_acc += v
    if v == 1.:
        classes_1.append(k)
    if v == 0.:
        classes_0.append(k)


avg_acc/200, len(classes_1), len(classes_0), classes_1[:5], classes_0[:5]


# %%
