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
from datasets import _transform, CUBDataset, NABirdsDataset, INaturalistDataset
from collections import OrderedDict
import clip

import cv2, pickle
from loading_helpers import *

from tqdm import tqdm

hparams = {}
# hyperparameters

hparams['model_size'] = "ViT-B/32" 
# Options:
# ['RN50',
#  'RN101',
#  'RN50x4',
#  'RN50x16',
#  'RN50x64',
#  'ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14',
#  'ViT-L/14@336px']
hparams['dataset'] = 'cub'

hparams['batch_size'] = 64*10
hparams['device'] = "cuda:4" if torch.cuda.is_available() else "cpu"
hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'

hparams['apply_descriptor_modification'] = True

hparams['verbose'] = False
hparams['image_size'] = 224
if hparams['model_size'] == 'ViT-L/14@336px' and hparams['image_size'] != 336:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.')
    hparams['image_size'] = 336
elif hparams['model_size'] == 'RN50x4' and hparams['image_size'] != 288:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 288
elif hparams['model_size'] == 'RN50x16' and hparams['image_size'] != 384:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 384
elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 448

hparams['before_text'] = ""
hparams['label_before_text'] = ""
hparams['between_text'] = ', '
# hparams['between_text'] = ' '
# hparams['between_text'] = ''
hparams['after_text'] = ''
hparams['unmodify'] = True
# hparams['after_text'] = '.'
# hparams['after_text'] = ' which is a type of bird.'
hparams['label_after_text'] = ''
# hparams['label_after_text'] = ' which is a type of bird.'
hparams['seed'] = 1

# TODO: fix this... defining global variable to be edited in a function, bad practice
# unmodify_dict = {}

# classes_to_load = openai_imagenet_classes
hparams['descriptor_fname'] = None

CUB_DIR = '/home/tin/datasets/cub/CUB/' # REPLACE THIS WITH YOUR OWN PATH
NABIRD_DIR = '/home/tin/datasets/nabirds/'
INATURALIST_DIR = '/home/tin/datasets/inaturalist2021_onlybird/'

# PyTorch datasets
tfms = _transform(hparams['image_size'])



class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, selected_folders, transform=None):
        self.data_dir = data_dir
        self.selected_folders = selected_folders
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for folder_name in self.selected_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            class_index = self.selected_folders.index(folder_name)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    data.append((file_path, class_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -------
if hparams['dataset'] == 'cub':
    # load CUB dataset
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    # dataset = ImageFolder(root='/home/tin/datasets/cub/CUB/flybird_cub_test/', transform=tfms)

    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'cub/descriptors_cub'

elif hparams['dataset'] == 'nabirds':
    hparams['data_dir'] = pathlib.Path(NABIRD_DIR)
    f = open("./descriptors/nabirds/no_ann_additional_chatgpt_descriptors_nabirds.json", "r")
    # f = open("./descriptors/nabirds/chatgpt_descriptors_nabirds.json", "r")
    data = json.load(f)
    subset_class_names = list(data.keys())

    # dataset = NABirdsDataset(hparams['data_dir'], train=False, subset_class_names=subset_class_names, transform=tfms)
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
    
    dataset = CustomImageDataset(data_dir='/home/tin/datasets/nabirds/flybird_nabirds_test/', selected_folders=selected_folders, transform=tfms)
    # dataset = ImageFolder(root='/home/tin/datasets/nabirds/test/', transform=tfms)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'nabirds/descriptors_nabirds'

elif hparams['dataset'] == 'inaturalist2021':
    hparams['data_dir'] = pathlib.Path(INATURALIST_DIR)
    f = open("./descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json", "r")
    data = json.load(f)
    subset_class_names = list(data.keys())
    dataset = INaturalistDataset(root_dir=hparams['data_dir'], train=False, subset_class_names=subset_class_names, n_pixel=hparams['image_size'], transform=tfms)
    
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_inaturalist2021'


    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors/others/descriptors_eurosat.json'

hparams['model_size'] = "ViT-B/32" 
if hparams["model_size"] == 'ViT-B/32':
    model_size = 'B32' 
elif hparams["model_size"] == 'ViT-B/16':
    model_size = 'B16'
else:
    model_size = 'L14'
hparams['device'] = "cuda:5" if torch.cuda.is_available() else "cpu"
hparams['descriptor_fname'] = './descriptors/' + hparams['descriptor_fname']

# imagenet
# hparams['descriptor_fname'] = './descriptors/others/descriptors_imagenet.json'

# cub
# hparams['descriptor_fname'] = f"./descriptors/cub/descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/cub/additional_sachit_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/cub/chatgpt_descriptors_{hparams['dataset']}.json"
hparams['descriptor_fname'] = f"./descriptors/cub/additional_chatgpt_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/cub/ID_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/cub/ID2_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/cub/gpt_4_sachit_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/hier_additional_chatgpt_descriptors_{hparams['dataset']}.json"

# nabirds
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_chatgpt_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_ID_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_ID2_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_additional_chatgpt_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_sachit_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/no_ann_additional_sachit_descriptors_{hparams['dataset']}.json"
# hparams['descriptor_fname'] = f"./descriptors/nabirds/additional_sachit_descriptors_{hparams['dataset']}.json"
# inaturalist
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_sachit_descriptors_inaturalist.json'
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json'
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json'
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_ID2_descriptors_inaturalist.json'
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_additional_sachit_descriptors_inaturalist.json'
# hparams['descriptor_fname'] = './descriptors/inaturalist2021/425_additional_chatgpt_descriptors_inaturalist.json'
    
print(hparams['descriptor_fname'])
print("Creating descriptors...")
sci2comm=None
if hparams['dataset'] == 'inaturalist2021':
    sci2comm_path = "/home/tin/projects/reasoning/plain_clip/sci2comm_inat_425.json"
    sci2comm = open(sci2comm_path, 'r')
    sci2comm = json.load(sci2comm)

gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load, sci_2_comm=None)
label_to_classname = list(gpt_descriptions.keys())
if sci2comm:
    label_to_classname = [sci2comm[i] for i in label_to_classname]

n_classes = len(list(gpt_descriptions.keys()))

# allaboutbirds_example_images_path = f'./image_descriptions/cub/allaboutbirds_example_images_40.json' # 30 is the best for cub
allaboutbirds_example_images_path = f'./image_descriptions/nabirds/nabirds_same_example_images_50.json' # 30 is the best
# allaboutbirds_example_images_path = f'./image_descriptions/inaturalist/inaturalist_example_images_50.json'
# allaboutbirds_example_images_path = f'./image_descriptions/imagenet/imagenet_example_images_500.json'
allaboutbirds_example_images = open(allaboutbirds_example_images_path, 'r')
allaboutbirds_example_images = json.load(allaboutbirds_example_images)

def compute_description_encodings(model):
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
    #         imgs = imgs.to(hparams['device'])
    #         description_encodings[k] = F.normalize(model.encode_image(imgs)).to('cpu')
    #     # save embs files
        
    #     output_filename = f'./pre_feats/{hparams["dataset"]}/{model_size}_no_ann_same_visual_encodings_reversed.npz'

    #     keys = list(description_encodings.keys())
    #     values = [description_encodings[key] for key in keys]
    #     np.savez(output_filename, **dict(zip(keys, values)))
    #     exit()

    # desired_order = list(gpt_descriptions.keys())
    # allaboutbirds_example_images = {k.lower(): v for k,v in allaboutbirds_example_images.items()}
    # allaboutbirds_example_images = {key: allaboutbirds_example_images[key] for key in desired_order}
    
    for k, v in gpt_descriptions.items():
        # v = v[:limited_descs]
        
        if len(v[-3]) >= cut_len:
            v[-3] = v[-3][:cut_len]
        if len(v[-2]) >= cut_len:
            v[-2] = v[-2][:cut_len]
        if len(v[-1]) >= cut_len:
            v[-1] = v[-1][:cut_len]
        
        if hparams['descriptor_fname'] in ["./descriptors/nabirds/no_ann_additional_sachit_descriptors_nabirds.json", \
                                           './descriptors/inaturalist2021/replaced_425_additional_sachit_descriptors_inaturalist.json',\
                                            './descriptors/inaturalist2021/425_additional_sachit_descriptors_inaturalist.json']:
            if len(v[-4]) >= cut_len:
                v[-4] = v[-4][:cut_len]    

        # if hparams['descriptor_fname'] in ["./descriptors/ID_descriptors_cub.json",
        #                                     "./descriptors/ID2_descriptors_cub.json",
        #                                     "./descriptors/ID_descriptors_nabirds.json",
        #                                     "./descriptors/ID2_descriptors_nabirds.json",
        #                                     "./descriptors/ID_diffshape_descriptors_nabirds.json",
        #                                     "./descriptors/ID2_diffshape_descriptors_nabirds.json",
        #                                     "./descriptors/no_ann_ID_descriptors_nabirds.json"]:
        # if len(v[0]) >= cut_len:
        #     v[0] = v[0][:cut_len]

        
        new_v = []# for testing gpt4+habitat
        for i in range(len(v)):
            # if i not in [len(v)-1, len(v)-2, len(v)-3]: # sachit
            if i not in [len(v)-2, len(v)-3]: # gpt4
                new_v.append(v[i])
        
        tokens = clip.tokenize(new_v).to(hparams['device'])
        description_encodings[k] = F.normalize(model.encode_text(tokens))
        # description_encodings[k] = model.encode_text(tokens)
    
    # loaded_data = np.load(f'./pre_feats/{hparams["dataset"]}/{model_size}_no_ann_visual_encodings.npz')
    # # for i, (k, image_paths) in enumerate(allaboutbirds_example_images.items()):
    # for k, v in gpt_descriptions.items():
    #     num=10
    #     np.random.seed(116)
    #     random_indices = np.random.choice(loaded_data[k].shape[0], num, replace=False) if loaded_data[k].shape[0] >= num else [i for i in range(loaded_data[k].shape[0])]
    #     random_vectors = loaded_data[k][random_indices]

    #     # description_encodings[k] = torch.Tensor(loaded_data[k][:num]).to(hparams['device'], dtype=torch.float16)
    #     img_feats = torch.Tensor(random_vectors).to(hparams['device'], dtype=torch.float16) # loaded_data[k][:num]
    #     description_encodings[k] = torch.cat([description_encodings[k], img_feats], dim=0)

        ##############
        # for num, p in enumerate(image_paths):
        #     if num == 40:
        #         break
        #     img = Image.open(p)
        #     imgs.append(tfms(img))
            
        # imgs = torch.stack(imgs)
        # imgs = imgs.to(hparams['device'])
        # description_encodings[k] = F.normalize(model.encode_image(imgs))
        # img_feats = model.encode_image(imgs)
        # if hparams['dataset'] == 'pet':
        #     description_encodings[k.lower()] = F.normalize(torch.cat([description_encodings[k.lower()], img_feats], dim=0))
        # else:
        #     description_encodings[k] = F.normalize(torch.cat([description_encodings[k], img_feats], dim=0))
            
    return description_encodings

def compute_label_encodings(model):
    label_encodings = F.normalize(model.encode_text(clip.tokenize([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
    return label_encodings


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    elif aggregation_method == 'random':
        # similarity_matrix_chunk[similarity_matrix_chunk < 0.3] = 0.
        similarity_matrix_chunk[:, 8:][similarity_matrix_chunk[:, 8:] < 0.8] = 0.
        # similarity_matrix_chunk = similarity_matrix_chunk[:, :8]
        return similarity_matrix_chunk.mean(dim=1)
    elif aggregation_method == 'weighted':
        alpha = 0.0
        similarity_matrix_chunk[:, :5] *= alpha
        similarity_matrix_chunk[:, 5:] *= 1-alpha
        return similarity_matrix_chunk.mean(dim=1)
    
    else: raise ValueError("Unknown aggregate_similarity")

def show_from_indices(indices, images, labels=None, predictions=None, predictions2 = None, n=None, image_description_similarity=None, image_labels_similarity=None):
    if indices is None or (len(indices) == 0):
        print("No indices provided")
        return
    
    if n is not None:
        indices = indices[:n]
    
    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")
        
        print("\n")
        
        if image_labels_similarity is not None:
            if labels is not None:
                print(f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name: 
                    print("Predicted label (ours) matches true label")
                else: 
                    print(f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else: 
                    print(f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")
        
            print("\n")
        
        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                    # continue
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label, predicted_label_name, "descriptor")
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                    # continue
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label2, predicted_label_name2, "CLIP")
            print("\n")

def print_descriptor_similarity(image_description_similarity, index, label, label_name, label_type="provided"):
    # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
        k = unmodify_dict[label_name][k]
        # print("\t" + f"matched \"{k}\" with score: {v}")
        print(f"{k}\t{100.*v}")
        
def print_max_descriptor_similarity(image_description_similarity, index, label, label_name):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")
    
def show_misclassified_images(images, labels, predictions, n=None, 
                              image_description_similarity=None, 
                              image_labels_similarity=None,
                              true_label_to_consider: int = None, 
                              predicted_label_to_consider: int = None):
    misclassified_indices = yield_misclassified_indices(images, labels=labels, predictions=predictions, true_label_to_consider=true_label_to_consider, predicted_label_to_consider=predicted_label_to_consider)
    if misclassified_indices is None: return
    show_from_indices(misclassified_indices, images, labels, predictions, 
                      n=n,
                      image_description_similarity=image_description_similarity, 
                      image_labels_similarity=image_labels_similarity)

def yield_misclassified_indices(images, labels, predictions, true_label_to_consider=None, predicted_label_to_consider=None):
    misclassified_indicators = (predictions.cpu() != labels.cpu())
    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)
    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)
        
    if misclassified_indicators.sum() == 0:
        output_string = 'No misclassified images found'
        if true_label_to_consider is not None:
            output_string += f' with true label {label_to_classname[true_label_to_consider]}'
        if predicted_label_to_consider is not None:
            output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
        print(output_string + '.')
            
        return
    
    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


from PIL import Image
def predict_and_show_explanations(images, model, labels=None, description_encodings=None, label_encodings=None, device=None):
    if type(images) == Image:
        images = tfms(images)
        
    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    n_classes = len(description_encodings)
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)