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

# %%
class cfg:
    dataset = 'inat21'#inat21, cub, nabirds
    batch_size = 8
    device = "cuda:4" if torch.cuda.is_available() else "cpu"

    CUB_DIR = '/home/tin/datasets/CUB_200_2011/'
    NABIRD_DIR = '/home/tin/datasets/nabirds/'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

    MODEL_TYPE = 'ViT-L/14'
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

elif cfg.dataset == 'nabirds':
    dataset_dir = pathlib.Path(cfg.NABIRD_DIR)
    f = open("./descriptors/nabirds/no_ann_additional_chatgpt_descriptors_nabirds.json", "r")
    data = json.load(f)
    subset_class_names = list(data.keys())
    dataset = NABirdsDataset(dataset_dir, train=False, subset_class_names=subset_class_names, transform=preprocess)
    
    def read_classes(bird_dir):
        """Loads DataFrame with class labels. Returns full class table
        and table containing lowest level classes.
        """
        def make_annotation(s):
            try:
                return s.split('(')[1].split(')')[0]
            except Exception as e:
                return None

        classes = pd.read_table(f'{bird_dir}/classes.txt', header=None)
        classes['id'] = classes[0].apply(lambda s: int(s.split(' ')[0]))
        classes['label_name'] = classes[0].apply(lambda s: ' '.join(s.split(' ')[1:]))
        classes['annotation'] = classes['label_name'].apply(make_annotation)
        classes['name'] = classes['label_name'].apply(lambda s: s.split('(')[0].strip())

        return classes

    idx2class_df = read_classes(bird_dir='/home/tin/datasets/nabirds/')
    nabirds_idx2class = idx2class_df.set_index('id')['name'].to_dict()

elif cfg.dataset == 'inat21':
    dataset_dir = pathlib.Path(cfg.INATURALIST_DIR)
    f = open("./descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json", "r")
    data = json.load(f)
    subset_class_names = list(data.keys())
    dataset = INaturalistDataset(root_dir=dataset_dir, train=False, subset_class_names=subset_class_names, transform=preprocess)

    bird_classes_file = '/home/tin/datasets/inaturalist2021_onlybird/bird_classes.json'
    f = open(bird_classes_file, 'r')
    data = json.load(f)
    idx2class = data['name']
    idx2imagedir = data['image_dir_name']
    inat21_imagedir2class = {v:idx2class[k] for k, v in idx2imagedir.items()}

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
# %%
print("Number of images: ", len(image_paths))
# %% test retrieving image by text
# text = "Laysan Albatrosses spend most of their time on the open Pacific Ocean, spanning tropical waters up to the southern Bering Sea"
text = "Laysan Albatross, Laysan Albatrosses nest on open, sandy or grassy islands, mostly in the Hawaiian Island chain"
returned_image_paths, text_after = find_image_by_text(text, image_features, image_paths, n=10)
print(f"Before: {text}")
print(f"After: {text_after}")

# %%
show_images(returned_image_paths)

# %% test retrieving image by image
image_path = 'test_bird.jpeg'
returned_image_paths = find_image_by_image(image_path, image_features, image_paths, n=5)
# %%
# show_images(returned_image_paths)
# %% --- get the habitat description ---
description_path = None
match cfg.dataset:
    case "cub":
        description_path = "./descriptors/cub/additional_chatgpt_descriptors_cub.json"
    case "nabirds":
        description_path = "./descriptors/nabirds/no_ann_additional_chatgpt_descriptors_nabirds.json"
    case "inat21":
        description_path = "./descriptors/inaturalist2021/425_additional_chatgpt_descriptors_inaturalist.json"

f = open(description_path, 'r')
data = json.load(f)
data = {k: v[-1][9:] for k,v in data.items()}
# split a sentence into multiple sentences
data = {k: v.split('.') for k,v in data.items()}
data = {k: [f'{k}, {s}' for s in v] for k,v in data.items()}
num_classes = len(data.keys())
# %%
avg_len = 0

for i, (k, v) in enumerate(data.items()):
    len_sub_sentences = len(v)
    avg_len += len_sub_sentences
    
avg_len/len(data)

# save data
# json_object = json.dumps(data, indent=4)
# with open(f"habitat_{description_path.split('/')[-1]}", "w") as f:
#     f.write(json_object)
# %% each class retrieves N images
import shutil, os
save_retrieved_path = f"retrieval_{cfg.dataset}_images_by_texts/"    
if not os.path.exists(save_retrieved_path):
    os.makedirs(save_retrieved_path)

retrieval_acc_dict = {}
retrieved_num = 5

for k, v in data.items():
    # v = v.replace(k, 'this bird')
    class_name = k.replace('-', ' ').lower() if cfg.dataset == 'cub' else k
    
    if class_name not in retrieval_acc_dict:
        retrieval_acc_dict[class_name] = 0

    if not os.path.exists(os.path.join(save_retrieved_path, k)):
        os.makedirs(os.path.join(save_retrieved_path, k))
    
    total_returned_image_paths = []
    v_after = []
    for s in v:
        returned_image_paths, s_after = find_image_by_text(s, image_features, image_paths, n=retrieved_num)
        total_returned_image_paths += returned_image_paths
        v_after.append(s_after)
    returned_image_paths = list(set(total_returned_image_paths))

    # save image and query
    for p in returned_image_paths:
        shutil.copy(p, os.path.join(save_retrieved_path, k))
        if cfg.dataset == 'cub':
            retrieved_image_class_name = p.split('/')[-1].split('_')[:-2]
            retrieved_image_class_name = " ".join(retrieved_image_class_name).lower()
        elif cfg.dataset == 'nabirds':
            retrieved_image_class_index = p.split('/')[-2]
            retrieved_image_class_index = int(retrieved_image_class_index)
            retrieved_image_class_name = nabirds_idx2class[retrieved_image_class_index]
        elif cfg.dataset == 'inat21':
            retrieved_imagedir = p.split('/')[-2]
            retrieved_image_class_name = inat21_imagedir2class[retrieved_imagedir]

        if retrieved_image_class_name == class_name:
            retrieval_acc_dict[class_name] += 1
    retrieval_acc_dict[class_name] /= len(returned_image_paths)

    with open(f'{os.path.join(save_retrieved_path, k)}/query.txt', 'w') as f:
        f.write('BEFORE: \n')
        for s in v:
            f.write(f'{s}\n')
        f.write('AFTER: \n')
        for s_after in v_after:
            f.write(f'{s_after}\n')

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

json_object = json.dumps(retrieval_acc_dict, indent=4)
with open(f'{cfg.dataset}_retrieval_acc.json', "w") as outfile:
    outfile.write(json_object)

100*(avg_acc/num_classes), len(classes_1), len(classes_0), classes_1[:5], classes_0[:5]

# %%
