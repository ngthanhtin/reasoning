import json
import numpy as np
import torch
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
import pathlib

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset, NABirdsDataset, INaturalistDataset, PartImageNetDataset, CustomImageDataset
from collections import OrderedDict
import clip

import cv2, pickle
from loading_helpers import *

from tqdm import tqdm

# List of methods available to use.
METHODS = [
    'clip',
    'clip_habitat',
    'gpt_descriptions',
    'waffle'
]

# List of compatible datasets.
DATASETS = [
    'part_imagenet',  
    'cub',
    'nabirds',
    'inaturalist'
]

# List of compatible backbones.
BACKBONES = [
    'ViT-B/32',
    'ViT-B/16',
    'ViT-L/14',    
]

def setup(opt):
    opt.image_size = 224
    if opt.model_size == 'ViT-L/14@336px' and opt.image_size != 336:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 336.')
        opt.image_size = 336
    elif opt.model_size == 'RN50x4' and opt.image_size != 288:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 288.')
        opt.image_size = 288
    elif opt.model_size == 'RN50x16' and opt.image_size != 384:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 384.')
        opt.image_size = 384
    elif opt.model_size == 'RN50x64' and opt.image_size != 448:
        print(f'Model size is {opt.model_size} but image size is {opt.image_size}. Setting image size to 448.')
        opt.image_size = 448

    CUB_DIR = '/home/tin/datasets/cub/CUB/'
    NABIRD_DIR = '/home/tin/datasets/nabirds/'
    INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'
    PART_IMAGENET_DIR = '/home/tin/datasets/PartImageNet/'

    # PyTorch datasets
    opt.tfms = _transform(opt.image_size)

    if opt.dataset == 'cub':
        # load CUB dataset
        opt.data_dir = pathlib.Path(CUB_DIR)
        dataset = CUBDataset(opt.data_dir, train=False, transform=opt.tfms)
        # dataset = ImageFolder(root='/home/tin/datasets/cub/CUB_bb_on_birds_test/', transform=tfms)

        opt.classes_to_load = None #dataset.classes
        opt.num_classes = 200

    elif opt.dataset == 'nabirds':
        opt.data_dir = pathlib.Path(NABIRD_DIR)
        f = open(opt.descriptor_fname, "r")
        # f = open("./descriptors/nabirds/chatgpt_descriptors_nabirds.json", "r")
        data = json.load(f)
        subset_class_names = list(data.keys())

        # dataset = NABirdsDataset(opt.data_dir, train=False, subset_class_names=subset_class_names, transform=opt.tfms)
        # use to test flybird non fly bird
        # dictionary mapping image folder name and the class name
        foldername_2_classname_dict = {}
        classname_2_foldername_dict = {}

        with open('/home/tin/datasets/nabirds/classes.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(' ', 1)
                key = parts[0]
                key = '0'*(4-len(key)) + key
                
                value = parts[1]
                foldername_2_classname_dict[key] = value
                classname_2_foldername_dict[value] = key
        selected_folders = []
        for k, v in classname_2_foldername_dict.items():
            if k in subset_class_names:
                selected_folders.append(v)
        selected_folders=sorted(selected_folders)
        
        dataset = CustomImageDataset(data_dir='/home/tin/datasets/nabirds/images/', selected_folders=selected_folders, transform=opt.tfms)
        # dataset = ImageFolder(root='/home/tin/datasets/nabirds/test/', transform=opt.tfms)
        opt.classes_to_load = None #dataset.classes
        opt.num_classes = 267

    elif opt.dataset == 'inaturalist':
        opt.data_dir = pathlib.Path(INATURALIST_DIR)
        f = open(opt.descriptor_fname, "r")
        data = json.load(f)
        subset_class_names = list(data.keys())
        dataset = INaturalistDataset(root_dir=opt.data_dir, train=False, subset_class_names=subset_class_names, n_pixel=opt.image_size, transform=opt.tfms)
        
        # scientific names to common names and vice versa
        if opt.sci2comm:
            sci2comm_path = "/home/tin/projects/reasoning/plain_clip/sci2comm_inat_425.json"
            opt.sci2comm = open(sci2comm_path, 'r')
            opt.sci2comm = json.load(opt.sci2comm)

        opt.classes_to_load = None #dataset.classes
        opt.num_classes = 425

    elif opt.dataset == 'part_imagenet':
        opt.data_dir = pathlib.Path(PART_IMAGENET_DIR)
        f = open(opt.descriptor_fname, "r")
        data = json.load(f)
        dataset = PartImageNetDataset(root_dir=opt.data_dir, description_dir=opt.descriptor_fname, train=False, n_pixel=opt.image_size, transform=opt.tfms)
        
        opt.classes_to_load = None #dataset.classes
        opt.num_classes = 158

    if opt.support_images_path:
        # support_images_path = f'./image_descriptions/cub/allaboutbirds_example_images_40.json' # 30 is the best for cub
        # support_images_path = f'./image_descriptions/nabirds/nabirds_same_example_images_50.json' # 30 is the best
        # support_images_path = f'./image_descriptions/inaturalist/inaturalist_example_images_50.json'
        
        support_images = open(opt.support_images_path, 'r')
        support_images = json.load(support_images)

    return opt, dataset



def compute_description_encodings(opt, model):
    print(f"Creating {opt.mode} descriptors...")
    gpt_descriptions, unmodify_dict = load_gpt_descriptions_2(opt, opt.classes_to_load, sci_2_comm=opt.sci2comm, mode=opt.mode)

    for k in gpt_descriptions:
        print(f"\nExample description for class {k}: \"{gpt_descriptions[k]}\"\n")
        break

    # global allaboutbirds_example_images  # Declare as global

    cut_len = 250 # 250
    limited_descs = 5
    description_encodings = OrderedDict()

    # save_visual_feat = False
    # if save_visual_feat:
    #     for i, (k, image_paths) in tqdm(enumerate(allaboutbirds_example_images.items())):
    #         imgs = []
    #         for ii, p in enumerate(image_paths[::-1]):
    #             # if ii > 100:
    #             #     break
    #             img = Image.open(p)
    #             imgs.append(tfms(img))
                    
    #         imgs = torch.stack(imgs)
    #         imgs = imgs.to(opt.device)
    #         description_encodings[k] = F.normalize(model.encode_image(imgs)).to('cpu')
    #     # save embs files
        
    #     output_filename = f'./pre_feats/{opt.dataset}/{model_size}_no_ann_same_visual_encodings_reversed.npz'

    #     keys = list(description_encodings.keys())
    #     values = [description_encodings[key] for key in keys]
    #     np.savez(output_filename, **dict(zip(keys, values)))
    #     exit()

    # desired_order = list(gpt_descriptions.keys())
    # allaboutbirds_example_images = {k.lower(): v for k,v in allaboutbirds_example_images.items()}
    # allaboutbirds_example_images = {key: allaboutbirds_example_images[key] for key in desired_order}
    # for k, v in gpt_descriptions.items():
    #     gpt_descriptions[k] = [k, gpt_descriptions[k][-1]] # classname and habitat

    for k, v in gpt_descriptions.items():
        # v = v[:limited_descs] # limit the number of descriptions per class
        v = [v_[:cut_len] for v_ in v] # limit the number of character per description
    
        tokens = clip.tokenize(v).to(opt.device)
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    
    # loaded_data = np.load(f'./pre_feats/{opt.dataset}/{model_size}_visual_encodings.npz')
    # # for i, (k, image_paths) in enumerate(allaboutbirds_example_images.items()):
    # for k, v in gpt_descriptions.items():
    #     num=16
    #     # random_indices = np.random.choice(loaded_data[k].shape[0], num, replace=False) if loaded_data[k].shape[0] >= num else [i for i in range(loaded_data[k].shape[0])]
    #     # random_vectors = loaded_data[k][random_indices]

    #     # description_encodings[k] = torch.Tensor(loaded_data[k][:num]).to(opt.device, dtype=torch.float16)
    #     img_feats = torch.Tensor(loaded_data[k][:num]).to(opt.device, dtype=torch.float16) # loaded_data[k][:num]
    #     description_encodings[k] = torch.cat([description_encodings[k], img_feats], dim=0)
       
    return description_encodings

def compute_label_encodings(opt, model):
    print("Creating label descriptors...")
    gpt_descriptions, unmodify_dict = load_gpt_descriptions_2(opt, opt.classes_to_load, sci_2_comm=opt.sci2comm, mode=opt.mode)

    label_to_classname = list(gpt_descriptions.keys())

    label_encodings = F.normalize(model.encode_text(clip.tokenize([opt.label_before_text + wordify(l) + opt.label_after_text for l in label_to_classname]).to(opt.device)))
    return label_encodings

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    elif aggregation_method == 'weighted':
        alpha = 0.0
        similarity_matrix_chunk[:, :5] *= alpha
        similarity_matrix_chunk[:, 5:] *= 1-alpha
        return similarity_matrix_chunk.mean(dim=1)
    
    else: raise ValueError("Unknown aggregate_similarity")

    






