# %%
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import os
import json
import sys
from tqdm import tqdm
import random
import time
import copy
from datetime import datetime

# %% config
class CFG:
    seed = 42
    dataset = 'cub'
    model_name = 'transfg' #mohammad, vit, transfg
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1486}
    bird_num_classes = dataset2num_classes[dataset]
    habitat_num_classes = dataset2num_classes[dataset]
    alpha = 1.

    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }

    # cutmix
    cutmix = False

    #hyper params
    lr = 1e-5 if model_name in {'vit', 'transfg'} else 1e-4
    image_size = 224 if model_name in {'mohammad', 'vit'} else 448
    image_expand_size = 256 if model_name in {'mohammad', 'vit'} else 600
    epochs = 100 if model_name in {'vit', 'transfg'} else 20

    # train or test
    train = True
    return_paths = not train
    batch_size = 64
    if model_name == 'transfg':
        batch_size = 8
    else:
        batch_size = 64 if train else 512

    # inat21
    inat21_df_path = 'inat21_onlybirds.csv'
    write_inat_to_df = not os.path.exists(inat21_df_path)

    # save folder
    save_folder    = f'./results/{dataset}_{model_name}_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    if not os.path.exists(save_folder) and train:
        os.makedirs(save_folder)

# Save the CFG instance
cfg_instance = CFG()
cfg_attributes = [attr for attr in dir(cfg_instance) if not callable(getattr(cfg_instance, attr)) and not attr.startswith("__")]
cfg_attributes_dict = {}
for attr in cfg_attributes:
    if attr == 'device':
        continue
    cfg_attributes_dict[attr] = getattr(cfg_instance, attr)

with open(f'{CFG.save_folder}/cfg_instance.json', 'w') as json_file:
    json.dump(cfg_attributes_dict, json_file, indent=4)
######
# %%
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

set_seed(seed=CFG.seed)

# %%
def Augment(train = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(CFG.image_expand_size),
            transforms.CenterCrop(CFG.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(CFG.image_expand_size),
            transforms.CenterCrop(CFG.image_expand_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform
    
class Unified_Inat21_Dataset(Dataset):
    def __init__(self, dataroot, df, transform=None, mode='train'):
        self.df = df
        self.df = self.df[self.df['Mode'] == mode]
        self.mode = mode
        self.transform = transform
        
        # cluster
        class_cluster_filepath = f"/home/tin/projects/reasoning/plain_clip/class_inat21_clusters_1486.json"
        f = open(class_cluster_filepath, 'r')
        idx2cluster = json.load(f)
        
        folderclasses = os.listdir(f"{dataroot}/bird_train/")
        folderclass2class = {}

        for folder_name in folderclasses:
                name_parts = folder_name.split('_')
                name = name_parts[-2] + ' ' + name_parts[-1]
                
                folderclass2class[folder_name] = name

        class2folderclass = {v:k for k,v in folderclass2class.items()}

        self.habitat_img_2_class = {}
        for idx, classes in idx2cluster.items():
            for cls in classes:
                # convert cls to folder class
                folderclass = class2folderclass[cls]
                habitat_image_paths = os.listdir(f"{dataroot}/inat21_inpaint_all/{folderclass}")
                for img_path in habitat_image_paths:
                    self.habitat_img_2_class[img_path] = int(idx) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        orig_image_path, label, mode = self.df.iloc[index].to_list()
        inpaint_image_path = orig_image_path.replace("bird_train", "inat21_inpaint_all")

        label = int(label)
        image = Image.open(orig_image_path).convert("RGB")
        inpaint_image = Image.open(inpaint_image_path).convert("RGB")
        label2 = self.habitat_img_2_class[inpaint_image_path.split('/')[-1]]

        if self.transform is not None:
            image = self.transform(image)
            inpaint_image = self.transform(inpaint_image)
        
        return (torch.cat((image, inpaint_image), 0), label, label2)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, return_paths=CFG.return_paths):
        super(ImageFolderWithPaths, self).__init__(root, transform, target_transform)
        self.root = root
        self.return_paths = return_paths

    def __getitem__(self, index):
        
        path, label = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_paths:
            return (img, label, path.split("/")[-2] + '/' + path.split("/")[-1])
        return (img, label)
    
def get_data_loaders(dataset, batch_size):
    """
    Get the train, val, test dataloader
    """
    if dataset in ['cub', 'nabirds']:
        # train
        orig_train_img_folder = 'CUB/train/'
        # orig_train_img_folder = 'CUB_augmix_train_small_2/'

        # test 
        # CUB_inpaint_all_test (onlybackground) vs CUB_nobirds_test (blackout-birds)
        orig_test_img_folder = 'CUB/test/'
        # orig_test_img_folder = 'CUB_no_bg_test/'

        
        orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{orig_train_img_folder}"
        orig_test_data_dir = f"{CFG.dataset2path[dataset]}/{orig_test_img_folder}"

        train_data = ImageFolderWithPaths(root=orig_train_data_dir, transform=Augment(train=True))
        test_data = ImageFolderWithPaths(root=orig_test_data_dir, transform=Augment(train=False))
        val_data = test_data

        train_data_len = len(train_data)
        valid_data_len = len(val_data)
        test_data_len = len(test_data)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        classes = train_data.classes
        bird_classes = classes
        return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes)

    return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes)
# %%
(train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes) = get_data_loaders(CFG.dataset, CFG.batch_size)
# %%
dataloaders = {
    "train":train_loader,
    "val": val_loader,
    "test": test_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len,
    "test": test_data_len
}
dataset_sizes

# %%
from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
    
class MultiTaskModel_3(nn.Module):
    def __init__(self, num_classes_task1=CFG.bird_num_classes, num_classes_task2=CFG.habitat_num_classes):
        super(MultiTaskModel_3, self).__init__()

        self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
        self.backbone1.load_state_dict(my_model_state_dict, strict=True)

        # Freeze backbone (for training only)
        for param in list(self.backbone1.parameters())[:-2]:
            param.requires_grad = False

        self.layer_norm = nn.LayerNorm(num_classes_task1)
    def forward(self, x):
        return self.backbone1(x)

class ViTBase16(nn.Module):
    def __init__(self, n_classes=200, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained)
            
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
# %%
# cut mix rand bbox
def rand_bbox(size, lam, to_tensor=True):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    #uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if to_tensor:
        bbx1 = torch.tensor(bbx1)
        bby1 = torch.tensor(bby1)
        bbx2 = torch.tensor(bbx2)
        bby2 = torch.tensor(bby2)

    return bbx1, bby1, bbx2, bby2

def cutmix_same_class(images, labels, alpha=0.4):
    batch_size = len(images)

    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    num_classes = len(np.unique(labels))
    
    indices_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    class_indices = [c_indices for c_indices in indices_by_class if len(c_indices) > 1]
    class_indices = [np.random.permutation(c_indices) for c_indices in class_indices]

    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)

    image_h, image_w, _ = images.shape[1:]  # Assuming image shape in (height, width, channels)

    mixed_images = images.copy()
    mixed_labels = labels.copy()

    for c_indices in class_indices:
        shuffled_indices = np.roll(c_indices, random.randint(1, len(c_indices) - 1))
        indices_pairs = zip(c_indices, shuffled_indices)

        for idx1, idx2 in indices_pairs:
            image1 = images[idx1]
            image2 = images[idx2]

            cx = np.random.randint(0, image_w)
            cy = np.random.randint(0, image_h)

            bbx1 = np.clip(int(cx - image_w * cut_rat / 2), 0, image_w)
            bby1 = np.clip(int(cy - image_h * cut_rat / 2), 0, image_h)
            bbx2 = np.clip(int(cx + image_w * cut_rat / 2), 0, image_w)
            bby2 = np.clip(int(cy + image_h * cut_rat / 2), 0, image_h)

            mixed_images[idx1, bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_h * image_w))
            mixed_labels[idx1] = lam * labels[idx1] + (1.0 - lam) * labels[idx2]

    return torch.tensor(mixed_images), torch.tensor(labels)#torch.tensor(mixed_labels)

# %%
def show_batch_cutmix_images(dataloader):
    for images,labels in dataloader:
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)
        images, labels = cutmix_same_class(images, labels, CFG.cutmix_beta)

        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

# %%
def train(trainloader, validloader, optimizer, criterion, scheduler, model, num_epochs = 10):
    
    best_acc = 0.
    for epoch in range(num_epochs):
        print("")
        model.train()
        train_loss, train_bird_acc = train_epoch(trainloader, model, criterion, optimizer)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Bird Acc: {train_bird_acc:.3f}")
        
        with torch.no_grad():    
            valid_loss, valid_bird_acc = evaluate_epoch(validloader, criterion, model)     
            print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss:.3f}, Valid Bird Acc: {valid_bird_acc:.3f}")
            # save model
            if best_acc <= valid_bird_acc:
                print("Saving...")
                best_acc = valid_bird_acc
                torch.save(model.state_dict(), f"{CFG.save_folder}/{epoch}-{best_acc:.3f}-cutmix_{CFG.cutmix}.pth")
        
            scheduler.step()
    
    return model

# %%
def train_epoch(trainloader, model, criterion, optimizer):
    model.train()
    losses = []
    bird_accs = []
    
    for inputs, bird_labels in tqdm(trainloader):
        # if CFG.cutmix and random.random() > 0.4:
        #     lam = np.random.beta(0.4, 0.4)
        #     rand_index = torch.randperm(inputs.size()[0])
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)    
        #     inputs[:, bbx1:bbx2, bby1:bby2, :] = inputs[rand_index, bbx1:bbx2, bby1:bby2, :]
            
        inputs = inputs.to(CFG.device)
        bird_labels = bird_labels.to(CFG.device)

        # zero the parameter gradients
        optimizer.zero_grad()
        bird_outputs = model(inputs)
    
        _, bird_preds = torch.max(bird_outputs, 1)
        loss = criterion(bird_outputs, bird_labels) 

        loss.backward()
        optimizer.step()

        # statistics
        losses.append(loss.item())
        bird_accs.append((torch.sum(bird_preds == bird_labels.data)/CFG.batch_size).detach().cpu().numpy())
            
    return np.mean(losses), np.mean(bird_accs)

# %%
def evaluate_epoch(validloader, criterion, model, return_paths=False):
    model.eval()
    losses = []
    bird_accs = []

    for inputs, bird_labels in tqdm(validloader):
        inputs = inputs.to(CFG.device)

        bird_outputs = model(inputs)

        bird_outputs = bird_outputs.detach().cpu() 
        
        _, bird_preds = torch.max(bird_outputs, 1)
        criterion = criterion.to('cpu')
        loss = criterion(bird_outputs, bird_labels) 
        criterion.to(CFG.device)

        # statistics
        losses.append(loss.item())
        bird_accs.append(torch.sum(bird_preds == bird_labels.data)/CFG.batch_size)
            
    return np.mean(losses), np.mean(bird_accs)


def test_epoch(testloader, model, return_paths=False):
    model.eval()
    full_paths = []
    running_corrects = 0

    for inputs, bird_labels, paths in tqdm(testloader):
        inputs = inputs.to(CFG.device)
        bird_labels = bird_labels.to(CFG.device)
        bird_outputs = model(inputs)

        _, preds = torch.max(bird_outputs, 1)
        probs, _ = torch.max(F.softmax(bird_outputs, dim=1), 1)
        running_corrects += torch.sum(preds == bird_labels.data)


    epoch_acc = running_corrects.double() / 5794#11788#5794

    print('-' * 10)
    print('Acc: {:.4f}'.format(100*epoch_acc))


# %%
from transfg.transfg_vit import VisionTransformer, CONFIGS
if CFG.model_name == 'mohammad':
    model = MultiTaskModel_3(num_classes_task1=200)
if CFG.model_name == 'vit':
    model = ViTBase16(n_classes=200, pretrained=True)
if CFG.model_name == 'transfg':
    # ViT from TransFG
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 448, zero_head=True, num_classes=200, smoothing_value=0.0)
    model.load_from(np.load("transfg/ViT-B_16.npz"))


model_params = model.parameters()
model.to(CFG.device)

criterion =  nn.CrossEntropyLoss()
criterion = criterion.to(CFG.device)

optimizer = optim.Adam(model_params, lr=CFG.lr)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)


if CFG.train:
    model_ft = train(train_loader, val_loader, optimizer, criterion, exp_lr_scheduler, model, num_epochs=CFG.epochs)
else:
    model.load_state_dict(torch.load("/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/transfg_cub_unified_vit_08_14_2023-14:53:56/34-0.890-cutmix_True.pth"))
    model.eval()
    with torch.no_grad():    
        test_epoch(test_loader, model, return_paths=CFG.return_paths)   