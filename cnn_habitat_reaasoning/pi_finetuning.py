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

import os, cv2
import json
import sys
from tqdm import tqdm
import random
import time
import copy
from datetime import datetime

# %% config
if not os.path.exists('results/'):
    os.makedirs('results/')
if not os.path.exists('results/cub/'):
    os.makedirs('results/cub/')
class CFG:
    seed = 45
    dataset = 'part_imagenet'
    model_name = 'transfg' #cnn or transfg
    use_cont_loss = True
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'part_imagenet': 158}
    
    # train, test data paths
    dataset2path = {
        # 'part_imagenet': '/home/tin/datasets/PartImageNet/images'
        'part_imagenet': '/home/tin/datasets/PartImageNet/aug_images'
    }
    orig_train_img_folder = 'augsame_train/' #'augsame_train/' #'train_folders/'
    orig_val_img_folder = 'val_folders/'
    orig_test_img_folder = 'test_folders/'
    
    #hyper params
    lr = 1e-5 if model_name in {'transfg'} else 1e-4
    image_size = 224 if model_name in {'cnn'} else 448
    image_expand_size = 256 if model_name in {'cnn'} else 600
    epochs = 50 if model_name in {'transfg'} else 20

    # train or test
    train = True
    return_paths = not train
    batch_size = 64
    if model_name == 'transfg':
        batch_size = 8
    else:
        batch_size = 64 if train else 512

    test_tta = False
    
    # save folder
    save_folder    = f'./results/{dataset}/{dataset}_single_{model_name}_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    if not os.path.exists(save_folder) and train:
        os.makedirs(save_folder)

# Save the CFG instance
if CFG.train:
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
            transforms.CenterCrop(CFG.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, return_paths=CFG.return_paths): #, num_images_per_class=3):
        super(ImageFolderWithPaths, self).__init__(root, transform, target_transform)
        self.root = root
        self.return_paths = return_paths

    #     if num_images_per_class != 0:
    #         self.num_images_per_class = num_images_per_class
    #         self._limit_dataset()

    # def _limit_dataset(self):
    #     new_data = []
    #     new_targets = []
    #     for class_idx in range(len(self.classes)):
    #         class_data = [item for item in self.samples if item[1] == class_idx]
    #         selected_samples = random.sample(class_data, min(self.num_images_per_class, len(class_data)))
    #         data, targets = zip(*selected_samples)
    #         new_data.extend(data)
    #         new_targets.extend(targets)
    #     self.samples = list(zip(new_data, new_targets))

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
    
    orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_train_img_folder}"
    orig_val_data_dir = f"/home/tin/datasets/PartImageNet/images/{CFG.orig_val_img_folder}"
    orig_test_data_dir = f"/home/tin/datasets/PartImageNet/images/{CFG.orig_test_img_folder}"

    train_data = ImageFolderWithPaths(root=orig_train_data_dir, transform=Augment(train=True)) #, num_images_per_class=0)
    val_data = ImageFolderWithPaths(root=orig_val_data_dir, transform=Augment(train=False)) #, num_images_per_class=3)
    test_data = ImageFolderWithPaths(root=orig_test_data_dir, transform=Augment(train=False)) #, num_images_per_class=3)

    train_data_len = len(train_data)
    valid_data_len = len(val_data)
    test_data_len = len(test_data)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = train_data.classes
    class_to_idx = test_data.class_to_idx
    return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes, class_to_idx)

# %%
(train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes, class_to_idx) = get_data_loaders(CFG.dataset, CFG.batch_size)
idx_to_class = {v:k for k,v in class_to_idx.items()}
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
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self, num_classes=158):
        super(CNNModel, self).__init__()

        self.backbone = models.resnet50(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.backbone(x) 

class ViTBase16(nn.Module):
    def __init__(self, n_classes=158, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained)
            
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# %%
def train(trainloader, validloader, optimizer, criterion, scheduler, model, num_epochs = 10):
    
    best_acc = 0.
    for epoch in range(num_epochs):
        print("")
        model.train()
        train_loss, train_acc = train_epoch(trainloader, model, criterion, optimizer)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
        
        with torch.no_grad():    
            valid_loss, valid_acc = evaluate_epoch(validloader, criterion, model)     
            print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")
            # save model
            if best_acc <= valid_acc:
                print("Saving...")
                best_acc = valid_acc
                torch.save(model.state_dict(), f"{CFG.save_folder}/{epoch}-{best_acc:.3f}.pth")
        
            scheduler.step()
    
    return model

# %%
def train_epoch(trainloader, model, criterion, optimizer):
    model.train()
    losses = []
    accs = []
    
    for inputs, labels in tqdm(trainloader):
        inputs = inputs.to(CFG.device)
        labels = labels.to(CFG.device)

        # zero the parameter gradients
        optimizer.zero_grad()
        if CFG.model_name == 'transfg' and CFG.use_cont_loss:
            loss, outputs = model(inputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels) 

        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()

        # statistics
        losses.append(loss.item())
        accs.append((torch.sum(preds == labels.data)/CFG.batch_size).detach().cpu().numpy())
            
    return np.mean(losses), np.mean(accs)

# %%
def evaluate_epoch(validloader, criterion, model, return_paths=False):
    model.eval()
    losses = []
    accs = []

    for inputs, labels in tqdm(validloader):
        inputs = inputs.to(CFG.device)
        
        outputs = model(inputs)

        outputs = outputs.detach().cpu() 
        
        _, preds = torch.max(outputs, 1)
        criterion = criterion.to('cpu')
        loss = criterion(outputs, labels) 
        criterion.to(CFG.device)

        # statistics
        losses.append(loss.item())
        accs.append(torch.sum(preds == labels.data)/CFG.batch_size)
            
    return np.mean(losses), np.mean(accs)

def test_epoch(testloader, model, return_paths=False):
    model.eval()
    full_paths = []
    running_corrects = 0

    class_corrects = [0] * 158
    class_counts = [0] * 158

    confusion_matrix = np.zeros((158, 158), dtype=int)

    for inputs, labels, paths in tqdm(testloader):
        inputs = inputs.to(CFG.device)
        labels = labels.to(CFG.device)
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
        running_corrects += torch.sum(preds == labels.data)

        for i in range(len(labels)):
            true_label = labels[i]
            predicted_label = preds[i]
            class_corrects[labels[i]] += int(preds[i] == labels[i])
            class_counts[labels[i]] += 1
            # Update the confusion matrix
            confusion_matrix[true_label][predicted_label] += 1

    class_accuracies = [correct / count if count != 0 else 0 for correct, count in zip(class_corrects, class_counts)]

    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    print('-' * 10)
    print('Acc: {:.4f}'.format(100*epoch_acc))

    return 100*epoch_acc, class_accuracies, confusion_matrix

def get_top_misclassified_classes(confusion_matrix, class_index, top_k=10):
    # Get the row corresponding to the class of interest
    class_row = confusion_matrix[class_index, :]
    
    # Exclude the correct classification count
    class_row[class_index] = 0
    
    # Get the indices of the top-k misclassified classes
    top_misclassified_indices = np.argsort(class_row)[::-1][:top_k]
    
    # Get the corresponding misclassification counts
    top_misclassified_counts = class_row[top_misclassified_indices]
    
    return top_misclassified_indices, top_misclassified_counts
# %%
from transfg.transfg_vit import VisionTransformer, CONFIGS
if CFG.model_name == 'cnn':
    model = CNNModel(num_classes=CFG.dataset2num_classes[CFG.dataset])
if CFG.model_name == 'transfg':
    # ViT from TransFG
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 448, zero_head=True, num_classes=CFG.dataset2num_classes[CFG.dataset], smoothing_value=0.0)
    model.load_from(np.load("transfg/ViT-B_16.npz"))

model.to(CFG.device)

criterion =  nn.CrossEntropyLoss()
criterion = criterion.to(CFG.device)

optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=0.1)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)


if CFG.train:
    print(CFG.orig_train_img_folder, CFG.orig_test_img_folder)
    model_ft = train(train_loader, val_loader, optimizer, criterion, exp_lr_scheduler, model, num_epochs=CFG.epochs)
else:
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_12_2024-15:00:39/16-0.895.pth'
    # mix-s
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-00:16:00/18-0.891.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:28:09/18-0.889.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:28:33/18-0.890.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:29:14/19-0.891.pth'

    # mix-g
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:07:38/7-0.897.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:30:15/19-0.899.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:30:49/12-0.898.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:31:16/9-0.899.pth'
    # mix-irr
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:08:38/19-0.897.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:31:59/18-0.897.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:32:24/13-0.899.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/part_imagenet_single_cnn_02_27_2024-02:32:53/10-0.900.pth'

    # baseline
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/transfg/normal/part_imagenet_single_transfg_02_27_2024-00:02:04/11-0.893.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/transfg/normal/part_imagenet_single_transfg_02_27_2024-10:27:33/23-0.892.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/transfg/normal/part_imagenet_single_transfg_02_27_2024-10:27:52/8-0.897.pth'

    # same
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/part_imagenet/transfg/augsame/part_imagenet_single_transfg_02_27_2024-00:52:34/6-0.897.pth'
    print(model_path)
    print(CFG.orig_test_img_folder)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
        
    
    with torch.no_grad():    
        acc, class_acc, confusion_matrix = test_epoch(test_loader, model, return_paths=CFG.return_paths)   
        top_misclassified_counts = 0
        
    