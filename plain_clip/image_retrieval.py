#%%
import os, sys, json
import natsort

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import random

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from tqdm import tqdm
from PIL import Image

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
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

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
    return F.normalize(model.encode_text(tokens)).detach().cpu().numpy()

def compute_image_feature(image):
    """
    a numpy image
    """
    pil_image = Image.fromarray(image)
    image = preprocess(pil_image)
    image = image.unsqueeze(0)
    features = model.encode_image(image)
    return F.normalize(features).detach().cpu().numpy()

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
    zeroshot_weights = compute_text_feature(text_query)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append(image_paths[idx])
    return file_paths

# %%
# image retrieval based on image-image
def find_image_by_image(image_query, image_features, image_paths, n=1):
    zeroshot_weights = compute_text_feature(text_query)
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

# %%
text = "lakes, rivers or ponds"
returned_image_paths = find_image_by_text(text, image_features, image_paths, n=3)
returned_image_paths

# %%
show_images(returned_image_paths)





# %% Name Entity Recognition
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# %%
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "pond, forest"
ner_results = nlp(example)
print(ner_results)
# %%
