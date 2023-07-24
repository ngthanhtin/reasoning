#%%
import os, sys, json, cv2
import natsort
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import random
import re

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from tqdm import tqdm
import pickle

import clip
import open_clip

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
    dataset = 'cub'#inat21, cub, nabirds
    retrieve_model = 'clip' #openclip, instructblip+sentence_transformers
    batch_size = 12
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    CUB_DIR = '/home/tin/datasets/cub/CUB_inpaint_all_train/'
    NABIRD_DIR = '/home/tin/datasets/nabirds/'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

    MODEL_TYPE = 'ViT-L/14'
    IMAGE_SIZE = 224

    # use additional data
    additional_datasets = []#['nabirds', 'inat21']

    # save image features path
    model_type = MODEL_TYPE.replace('/', '_')
    additional_dataset_names = '_'.join(additional_datasets)
    image_features_filename = f"./embeddings/{dataset}_{additional_dataset_names}_{retrieve_model}_{model_type}_image_features.pkl"
    image_paths_filename = f"./embeddings/{dataset}_{additional_dataset_names}_{retrieve_model}_{model_type}_image_paths.txt"

    # retrieve
    retrieved_num = 10
    save_retrieved_path = f"retrieved_{dataset}_{additional_dataset_names}/"    


# %%
from torchvision.datasets import ImageFolder
class ImageFolderWithPaths(ImageFolder):

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        
        return (img, label ,path)
# %%
# init CLIP
# load model (currently clip) to get box-query scores
def load_model(model_name, device):
    if model_name in clip.clip._MODELS:
        model, transform = clip.load(model_name, device=device, jit=False)
        tokenizer = clip.tokenize
    elif 'laion' in model_name:
        # from huggingface, the model card name has the following format: laion/CLIP-ViT-L-14-laion2B-s32B-b82K
        # where VIT-L-14 is the base model name, and laion2B-s32B-b82K is the training config
        pattern = r"(.*/)(.*?)-(.*?)-(.*?)-(.*?)-(.*)"
        matches = re.match(pattern, model_name)
        if matches:
            base_model_name = '-'.join(matches.group(3,4,5))
            training_config = matches.group(6)
        else:
            raise ValueError(f"model_name {model_name} is not in the correct format")
        model, training_transform, transform = open_clip.create_model_and_transforms(base_model_name, pretrained=training_config, device=device)
        tokenizer = open_clip.get_tokenizer(base_model_name)
    
    return model, transform, tokenizer

model, preprocess, tokenizer = load_model(cfg.MODEL_TYPE, device=cfg.device)
# %%
# create dataset and dataloder    
if cfg.dataset == 'cub':
    # load CUB dataset
    dataset_dir = pathlib.Path(cfg.CUB_DIR)
    dataset = ImageFolderWithPaths(cfg.CUB_DIR, transform=preprocess)

    # dataset = CUBDataset(dataset_dir, train=True, transform=preprocess)

elif cfg.dataset == 'nabirds':
    dataset_dir = pathlib.Path(cfg.NABIRD_DIR)
    f = open("./descriptors/nabirds/no_ann_additional_chatgpt_descriptors_nabirds.json", "r")
    data = json.load(f)
    subset_class_names = list(data.keys())
    dataset = NABirdsDataset(dataset_dir, train=True, subset_class_names=subset_class_names, transform=preprocess)
    
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

    tokens = tokenizer(desc).to(cfg.device)
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
    
    # add unsplash
    if len(cfg.additional_datasets) > 0:
        for dataset_type in cfg.additional_datasets:
            if dataset_type == 'nabirds':
                ext_data_path = '/home/tin/datasets/nabirds/nabirds_inpaint_kp_full/'
            elif dataset_type == 'inat21':
                ext_data_path = '/home/tin/datasets/inaturalist2021_onlybird/inat21_inpaint_all/'
            ext_dataset = ImageFolderWithPaths(ext_data_path,transform=preprocess)
            ext_dataloader = DataLoader(ext_dataset, cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True)

            for i, batch in enumerate(tqdm(ext_dataloader)):
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
if os.path.exists(cfg.image_features_filename) and os.path.exists(cfg.image_paths_filename):
    with open(cfg.image_features_filename, 'rb') as f:
        image_features = pickle.load(f)
        image_features = torch.tensor(image_features)
    with open(cfg.image_paths_filename, "r") as f:
        lines = f.readlines()
        image_paths = [line.replace("\n", "") for line in lines]
else:
    image_features, image_paths = compute_image_features(dataloader)
    with open(cfg.image_features_filename, "wb") as f:
        pickle.dump(image_features, f)
    with open(cfg.image_paths_filename, "w") as f:
        for p in image_paths:
            f.write(f"{p}\n") 

# %%
print("Number of images: ", len(image_paths))
# %% test retrieving image by text
# text = "Laysan Albatrosses spend most of their time on the open Pacific Ocean, spanning tropical waters up to the southern Bering Sea"
text = "Laysan Albatross, Laysan Albatrosses nest on open, sandy or grassy islands, mostly in the Hawaiian Island chain"
returned_image_paths, text_after = find_image_by_text(text, image_features, image_paths, n=4)
print(f"Before: {text}")
print(f"After: {text_after}")

# %%
show_images(returned_image_paths)
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
# data = {k: [s.lower().replace(k.lower(), '') for s in v] for k,v in data.items()}
if cfg.dataset == 'inat21':
    f = open('sci2comm_inat_425.json', 'r')
    sci2real_dict = json.load(f)
    sci2real_dict = {k:v.replace('_', '') for k, v in sci2real_dict.items()}
    data = {k: [f'{sci2real_dict[k]}, {s}' for s in v] for k,v in data.items()}

num_classes = len(data.keys())
data
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
retrieved_num = cfg.retrieved_num
save_retrieved_path = cfg.save_retrieved_path

if not os.path.exists(save_retrieved_path):
    os.makedirs(save_retrieved_path)

retrieval_acc_dict = {}


for k, v in data.items():
    # v = v.replace(k, 'this bird')
    class_name = k.replace('-', ' ').lower() if cfg.dataset == 'cub' else k
    
    if class_name not in retrieval_acc_dict:
        retrieval_acc_dict[class_name] = 0

    if not os.path.exists(os.path.join(save_retrieved_path, k)):
        os.makedirs(os.path.join(save_retrieved_path, k))
    
    total_returned_image_paths = []
    v_after = {}
    for i, s in enumerate(v):
        if i >= 2:
            break
        returned_image_paths, s_after = find_image_by_text(s, image_features, image_paths, n=retrieved_num)
        total_returned_image_paths += returned_image_paths
        v_after[s_after] = returned_image_paths
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

    with open(f'{os.path.join(save_retrieved_path, k)}/query.json', "w") as outfile:
        json.dump(v_after, outfile, indent=4)

    # with open(f'{os.path.join(save_retrieved_path, k)}/query.txt', 'w') as f:
    #     f.write('BEFORE: \n')
    #     for s in v:
    #         f.write(f'{s}\n')
    #     f.write('AFTER: \n')
    #     for s_after in v_after:
    #         f.write(f'{s_after}\n')

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
with open(f'{cfg.dataset}_retrieve_acc.json', "w") as outfile:
    outfile.write(json_object)

100*(avg_acc/num_classes), len(classes_1), len(classes_0), classes_1[:5], classes_0[:5]

# %% fix classname to idex.classname
import os
from shutil import move

path_to_fix = cfg.save_retrieved_path
path_ref = '/home/tin/datasets/cub/CUB/images/'

classname_idx = {}

list_class_fix = os.listdir(path_to_fix)
list_class_ref = os.listdir(path_ref)

for cls in list_class_ref:
    idx, classname = cls.split('.')
    # idx = int(idx)
    classname = classname.replace('_', ' ')
    split_key = classname.split(' ')
    if len(split_key) > 2: 
        classname = '-'.join(split_key[:-1]) + " " + split_key[-1]
    classname_idx[classname] = idx

for cls in list_class_fix:
    idx = classname_idx[cls]
    
    orig_path = f'{path_to_fix}/{cls}'
    cls = cls.replace(' ', '_')
    changed_path = f'{path_to_fix}/{idx}.{cls}'

    move(orig_path, changed_path)

# %% make data with no query.txt
path_to_fix = '/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts_inpaint_unsplash_query/'
new_path = '/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts_inpaint_unsplash/'

folders = os.listdir(path_to_fix)
for cls in folders:
    img_paths = os.listdir(os.path.join(path_to_fix, cls))
    src_paths = [f"{path_to_fix}/{cls}/{p}" for p in img_paths if 'txt' not in p]
    dest_paths = [f"{new_path}/{cls}/{p}" for p in img_paths if 'txt' not in p]
    if not os.path.exists(f"{new_path}/{cls}"):
        os.makedirs(f"{new_path}/{cls}")
    
    for src, dest in zip(src_paths, dest_paths):
        shutil.copy(src, dest)

# %% Python code snippet that counts the occurrences of image samples within folders under the root directory
import os
from collections import defaultdict

root_directory = cfg.save_retrieved_path  # Specify the root directory

image_counts = defaultdict(int)
total_images = 0
for dirpath, dirnames, filenames in os.walk(root_directory):
    total_images += len(filenames)-1
    for filename in filenames:
        if filename.lower().endswith('.jpg'):
            image_counts[filename] += 1

num_ = 0
deleted_paths = []
for filename, count in image_counts.items():
    if count >= 5:
        print(f'{filename}: {count} occurrences')
        num_ += 1
        
        # delete those repeated files
        for label_folder in os.listdir(root_directory):
            label_folder_path = f"{root_directory}/{label_folder}"
            if filename in os.listdir(label_folder_path):
                deleted_paths.append(f"{root_directory}/{label_folder}/{filename}")
                
for p in deleted_paths:
    os.remove(p)

for label_folder in os.listdir(root_directory):
    label_folder_path = f"{root_directory}/{label_folder}"
    if len(os.listdir(label_folder_path)) <= 2:
        print(label_folder)
num_, total_images, num_-len(deleted_paths)
# %% 

