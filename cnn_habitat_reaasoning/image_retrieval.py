from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import re

import torch
from torch.nn import functional as F

import pickle
from PIL import Image

import clip

import time

def load_model(model_name, device):
    model, transform = clip.load(model_name, device=device, jit=False)
    tokenizer = clip.tokenize

    
    return model, transform, tokenizer

# %%
def compute_image_feature(image, model, preprocess, device):
    """input: array: (W, H, C)"""
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)
    image_feat = model.encode_image(image).to(device)
    return F.normalize(image_feat).detach().cpu().numpy()

# %%
# image retrieval based on image-image
def load_features_and_paths():
    # load clip image features
    image_features_filename = "/home/tin/projects/reasoning/plain_clip/embeddings/orig_cub__clip_ViT-L_14_image_features.pkl"
    image_paths_filename = "/home/tin/projects/reasoning/plain_clip/embeddings/orig_cub__clip_ViT-L_14_image_paths.txt"
    with open(image_features_filename, 'rb') as f:
        image_features = pickle.load(f)
        image_features = torch.tensor(image_features)
    with open(image_paths_filename, "r") as f:
        lines = f.readlines()
        image_paths = [line.replace("\n", "") for line in lines]
    
    return image_features, image_paths

def find_image_by_image(model, preprocess, image_features, image_paths, image_path, device, n=1):
    image = Image.open(image_path)
    zeroshot_weights = compute_image_feature(image, model, preprocess, device)
    distances = np.dot(image_features, zeroshot_weights.reshape(-1, 1))
    file_paths = []
    for i in range(1, n+1):
        idx = np.argsort(distances, axis=0)[-i, 0]
        file_paths.append(image_paths[idx])

    return file_paths

def show_images(image_list):
    for im_path in image_list:
        print(im_path)
        image = Image.open(im_path)
        plt.imshow(image)
        plt.show()



# %% test retrieving image by image
# test_img_path = '/home/tin/datasets/cub/CUB/test/001.Black_footed_Albatross/Black_Footed_Albatross_0090_796077.jpg'
# returned_image_paths = find_image_by_image(test_img_path, n=4, device='cuda:1')

# print(returned_image_paths)
# %%
# show_images(returned_image_paths)