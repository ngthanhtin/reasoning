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
    'waffle',
    'waffle_habitat',
    'waffle_habitat_only'
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

    if opt.compute_support_images_embedding:        
        if opt.dataset == 'cub':
            support_images_json_path = '/home/tin/projects/reasoning/plain_clip/image_descriptions/cub/allaboutbirds_example_images_40.json'
        elif opt.dataset == 'nabirds':
            support_images_json_path = '/home/tin/projects/reasoning/plain_clip/image_descriptions/nabirds/nabirds_example_images_50.json'
        elif opt.dataset == 'inaturalist':
            support_images_json_path = '/home/tin/projects/reasoning/plain_clip/image_descriptions/inaturalist/inaturalist_example_images_50.json'
        elif opt.dataset == 'part_imagenet':
            support_images_json_path = '/home/tin/projects/reasoning/plain_clip/image_descriptions/part_imagenet/support_images.json'
        
        # dict: classname: paths
        support_images_dict = open(support_images_json_path, 'r')
        opt.support_images_dict = json.load(support_images_dict)

    return opt, dataset

def compute_description_encodings(opt, model):
    print(f"Creating {opt.mode} descriptors...")
    gpt_descriptions, unmodify_descriptions = load_gpt_descriptions_2(opt, opt.classes_to_load, sci_2_comm=opt.sci2comm, mode=opt.mode)

    for k in gpt_descriptions:
        print(f"\nExample description for class {k}: \"{gpt_descriptions[k]}\"\n")
        break

    cut_len = 250 # 250
    limited_descs = 5
    description_encodings = OrderedDict()

    if opt.compute_support_images_embedding:
        if opt.model_size == "ViT-B/32":
            output_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/B32_visual_encodings.npz'
        if opt.model_size == "ViT-B/16":
            output_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/B16_visual_encodings.npz'
        if opt.model_size == "ViT-L/14":
            output_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/L14_visual_encodings.npz'
        
        if os.path.exists(output_filename):
            print("The support images embedding is existed")
        else:
            print("Computing support images embedding...")
            from PIL import Image
            for i, (k, image_paths) in tqdm(enumerate(opt.support_images_dict.items())):
                imgs = []
                for ii, p in enumerate(image_paths):
                    img = Image.open(p)
                    imgs.append(opt.tfms(img))
                        
                imgs = torch.stack(imgs)
                imgs = imgs.to(opt.device)
                description_encodings[k] = F.normalize(model.encode_image(imgs)).to('cpu')
            
            # save embs files
            os.makedirs(f'./pre_feats/{opt.dataset}', exist_ok=True)

            keys = list(description_encodings.keys())
            values = [description_encodings[key] for key in keys]
            np.savez(output_filename, **dict(zip(keys, values)))

    for k, v in gpt_descriptions.items():
        v = [v_[:cut_len] for v_ in v] # limit the number of character per description
        
        tokens = clip.tokenize(v, truncate=True).to(opt.device)
        
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    
    if opt.use_support_images_embedding:
        if opt.model_size == "ViT-B/32":
            support_image_embedding_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/B32_visual_encodings.npz'
        if opt.model_size == "ViT-B/16":
            support_image_embedding_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/B16_visual_encodings.npz'
        if opt.model_size == "ViT-L/14":
            support_image_embedding_filename = f'/home/tin/projects/reasoning/plain_clip/pre_feats/{opt.dataset}/L14_visual_encodings.npz'

        if not os.path.exists(support_image_embedding_filename):
            print("Can not find the support images embedding...Use text embedding only...")
        else:
            print("Loading support images embedding...")
            loaded_data = np.load(support_image_embedding_filename)

            if opt.num_support_images == 1000: # use all the support images
                for k, v in gpt_descriptions.items():    
                    full_support_image_size = len(loaded_data[k])
                    break
                opt.num_support_images = full_support_image_size

            for i, (k, v) in enumerate(gpt_descriptions.items()):
                if opt.dataset == 'inaturalist' and opt.sci2comm:
                    comm_name = list(unmodify_descriptions.keys())[i]
                    img_feats = torch.Tensor(loaded_data[comm_name][:opt.num_support_images]).to(opt.device, dtype=torch.float16)
                else:
                    img_feats = torch.Tensor(loaded_data[k][:opt.num_support_images]).to(opt.device, dtype=torch.float16)
                # no need to normalize img_feats, because it was normalized at the time of precalculation
                description_encodings[k] = torch.cat([description_encodings[k], img_feats], dim=0)
       
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

    






