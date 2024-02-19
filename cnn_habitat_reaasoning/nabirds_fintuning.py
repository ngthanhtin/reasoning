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
if not os.path.exists('results/nabirds/'):
    os.makedirs('results/nabirds/')
class CFG:
    seed = 42
    dataset = 'nabirds' 
    model_name = 'transfg' # vit, mohammad, transfg
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    use_cont_loss = True

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1486}
    bird_num_classes = dataset2num_classes[dataset]
    # train, test data paths
    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
    }
    
    orig_train_img_folder = 'gen_data/augmix_images_small_diff_30_added_1/'#'gen_data/augmix_images_small_diff_30_added_1/' # 'train/', 'augirrelevant_images_small', 'augmix_images_small', 'augsame_images_small', gen_data/temp_gen_data/augirrelevant_with_orig_birds_train_60/
    #'gen_data/inpaint_images/test_inpaint/', 'gen_data/onlybird_images_test/', 'test/', 'gen_data/bb_on_birds_test/', 'gen_data/big_bb_on_birds_test/'
    orig_test_img_folder = 'test/'

    # orig_test_img_folder = 'gen_data/inpaint_images/test_inpaint/'
    # orig_test_img_folder = 'gen_data/inpaint_images/test_inpaint_non_flybird/test_inpaint/'
    # orig_test_img_folder = 'gen_data/big_bb_on_birds_non_flybird_test/big_bb_on_birds_test/'

    # orig_test_img_folder = 'gen_data/onlybird_images_test/'
    # orig_test_img_folder = 'gen_data/bb_on_birds_test/' 
    # orig_test_img_folder = 'gen_data/big_bb_on_birds_test/' 
    # # test with inat
    # orig_test_img_folder = '../overlapping_nabirds_inat_test/'
    # # # #test fly-nonfly birds
    # orig_test_img_folder = 'non_flybird_nabirds_test/'
    # orig_test_img_folder = 'flybird_nabirds_test/'

    # cutmix
    cutmix = False

    #hyper params
    lr = 1e-5 if model_name in {'vit', 'transfg'} else 1e-4
    image_size = 224 if model_name in {'mohammad', 'vit'} else 448
    image_expand_size = 256 if model_name in {'mohammad', 'vit'} else 600
    epochs = 50 if model_name in {'vit', 'transfg'} else 20 #15

    # train or test
    train = False

    return_paths = not train
    batch_size = 64
    if model_name == 'transfg':
        batch_size = 8
    else:
        batch_size = 64 if train else 512

    # save folder
    save_folder    = f'./results/{dataset}/{dataset}_single_{model_name}_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    if not os.path.exists(save_folder) and train:
        os.makedirs(save_folder)

if CFG.train:
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
    def __init__(self, root, transform=None, target_transform=None, return_paths=CFG.return_paths):#, num_images_per_class=3):
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
    if dataset in ['cub', 'nabirds']:
        orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_train_img_folder}"
        orig_test_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_test_img_folder}"

        train_data = ImageFolderWithPaths(root=orig_train_data_dir, transform=Augment(train=True))#, num_images_per_class=0)
        test_data = ImageFolderWithPaths(root=orig_test_data_dir, transform=Augment(train=False))#, num_images_per_class=0)
        val_data = test_data

        train_data_len = len(train_data)
        valid_data_len = len(val_data)
        test_data_len = len(test_data)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        classes = train_data.classes
        bird_classes = classes
        class_to_idx = test_data.class_to_idx
        return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes, class_to_idx)

# %%
(train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, bird_classes, class_to_idx) = get_data_loaders(CFG.dataset, CFG.batch_size)
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
from visual_correspondence_XAI.ResNet50.CUB_iNaturalist_17.FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
    
class MultiTaskModel_3(nn.Module):
    def __init__(self, num_classes_task=CFG.bird_num_classes):
        super(MultiTaskModel_3, self).__init__()

        self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth', map_location=torch.device('cpu'))
        self.backbone1.load_state_dict(my_model_state_dict, strict=True)

        # Freeze backbone (for training only)
        for param in list(self.backbone1.parameters())[:-2]:
            param.requires_grad = False

        self.branch = nn.Sequential(
            nn.Linear(200, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes_task)
        )
    
    def forward(self, x):
        features = self.backbone1(x)
        features = torch.relu(features)
        features = self.branch(features)
        
        return features

class ViTBase16(nn.Module):
    def __init__(self, n_classes=200, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224_in21k", pretrained=pretrained)
            
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x[:,:3,:,:])
        return x
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
        if CFG.model_name == 'transfg' and CFG.use_cont_loss:
            loss, bird_outputs = model(inputs, bird_labels)
        else:
            bird_outputs = model(inputs)
            loss = criterion(bird_outputs, bird_labels)
    
        _, bird_preds = torch.max(bird_outputs, 1)

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

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings, labels, graph_name, method='t-SNE', num_classes=555):
    if method == 't-SNE':
        reducer = TSNE(n_components=3)
    elif method == 'PCA':
        reducer = PCA(n_components=10)
    else:
        raise ValueError("Unsupported dimensionality reduction method")

    # Combine embeddings into a single numpy array
    embeddings = np.vstack(embeddings)
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Scatter plot the reduced embeddings with different colors for each class
    plt.figure(figsize=(20, 20))
    for i in range(num_classes):
        indices = np.where(labels == i)[0]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=f'Class {i}')
    plt.title(f'{method} Visualization of Embeddings')
    plt.legend()
    plt.savefig(f"{graph_name}_{CFG.model_name}_{CFG.dataset}_{method}_finetune.png")

def test_epoch(testloader, model, return_paths=False):
    model.eval()
    running_corrects = 0
    
    class_corrects = [0] * CFG.bird_num_classes
    class_counts = [0] * CFG.bird_num_classes

    true_labels = []
    predicted_labels = []
    embeddings = []  # Store the embeddings here

    for inputs, bird_labels, paths in tqdm(testloader):
        inputs = inputs.to(CFG.device)
        bird_labels = bird_labels.to(CFG.device)
        bird_outputs = model(inputs)

        _, preds = torch.max(bird_outputs, 1)
        probs, _ = torch.max(F.softmax(bird_outputs, dim=1), 1)
        running_corrects += torch.sum(preds == bird_labels.data)

        true_labels.extend(bird_labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

        for i in range(len(bird_labels)):
            class_corrects[bird_labels[i]] += int(preds[i] == bird_labels[i])
            class_counts[bird_labels[i]] += 1

        # Extract and store the embeddings
        embeddings.append(bird_outputs.cpu().numpy())

    class_accuracies = [correct / count if count != 0 else 0 for correct, count in zip(class_corrects, class_counts)]

    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    print('-' * 10)
    print('Acc: {:.4f}'.format(100*epoch_acc))

    return 100*epoch_acc, class_accuracies, embeddings, true_labels, predicted_labels

# %%
from transfg.transfg_vit import VisionTransformer, CONFIGS
if CFG.model_name == 'mohammad':
    model = MultiTaskModel_3(num_classes_task=555)
if CFG.model_name == 'vit':
    model = ViTBase16(n_classes=555, pretrained=True)
if CFG.model_name == 'transfg':
    # ViT from TransFG
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 448, zero_head=True, num_classes=555, smoothing_value=0.0)
    model.load_from(np.load("transfg/ViT-B_16.npz"))
# ##

model.to(CFG.device)

criterion =  nn.CrossEntropyLoss()
criterion = criterion.to(CFG.device)

optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
# optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=0.1)

if CFG.train:
    print(CFG.orig_train_img_folder)
    print(CFG.orig_test_img_folder)
    model_ft = train(train_loader, val_loader, optimizer, criterion, exp_lr_scheduler, model, num_epochs=CFG.epochs)
else:
    # orig, same, mix, irrelevant
    # mohammad
    # model_path = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/FINETUNE_nabirds_single_mohammad_08_14_2023-18:27:21/17-0.802-cutmix_False.pth" # finetune
    # model_path = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/SAME_nabirds_single_mohammad_08_15_2023-00:04:31/18-0.806-cutmix_False.pth" # augsame
    # model_path = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/MIX_nabirds_single_mohammad_08_15_2023-00:10:47/18-0.807-cutmix_False.pth" # augmix
    # model_path = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/60_BIRD_ORIG_IRRELEVANT_nabirds_single_mohammad_08_21_2023-01:21:45/19-0.792-cutmix_False.pth" # augirrelevant
    
    # transfg
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/FINETUNE_nabirds_single_transfg_08_17_2023-07:56:43/49-0.884-cutmix_False.pth' # finetune nabirds only
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/60_BIRD_ORIG_IRRELEVANT_nabirds_single_transfg_08_21_2023-01:22:50/23-0.877-cutmix_False.pth' # aug_irrelevant with orig birds
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/MIX_nabirds_single_transfg_08_16_2023-14:29:12/48-0.888-cutmix_False.pth' # aug_mix
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/SAME_nabirds_single_transfg_08_16_2023-01:08:53/31-0.886-cutmix_False.pth' # aug_same
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/MIX_2_nabirds_single_transfg_09_05_2023-19:59:30/27-0.886-cutmix_False.pth'

    # test
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_09_04_2023-20:41:35/15-0.807-cutmix_False.pth'
    # model_path = './results/nabirds/mohammad/SAME_2_nabirds_single_mohammad_09_05_2023-00:07:11/12-0.807-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/MIX_2_nabirds_single_mohammad_09_05_2023-14:39:15/19-0.810-cutmix_False.pth'

    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/SAME_2_nabirds_single_transfg_09_05_2023-16:02:00/34-0.887-cutmix_False.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/FINETUNE_nabirds_single_mohammad_08_14_2023-18:27:21/17-0.802-cutmix_False.pth'
    
    # mix
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-11:18:57/17-0.808-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-11:32:23/17-0.808-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-11:33:53/18-0.807-cutmix_False.pth'

    #irrelevant
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:51:47/18-0.810-cutmix_False.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:52:22/19-0.809-cutmix_False.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:52:58/18-0.810-cutmix_False.pth'

    # same
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:40:22/19-0.807-cutmix_False.pth'
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:40:53/17-0.809-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:41:15/18-0.806-cutmix_False.pth'

    # normal
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:36:52/19-0.806-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:37:05/17-0.806-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_mohammad_02_12_2024-13:37:17/19-0.807-cutmix_False.pth'

    # transfg

    # normal
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:19:09/35-0.887-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:19:32/44-0.883-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:19:50/46-0.884-cutmix_False.pth'
    # same
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:22:15/30-0.884-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:22:46/35-0.884-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_17_2024-16:23:21/30-0.885-cutmix_False.pth'
    # mix
    # model_path = ''
    # model_path = ''
    # model_path = ''
    #irrelevant
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_16_2024-08:29:51/13-0.878-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_16_2024-08:30:10/34-0.877-cutmix_False.pth'
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/nabirds_single_transfg_02_16_2024-08:30:30/28-0.879-cutmix_False.pth'

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(model_path)
    print(CFG.orig_test_img_folder)
    # write result to file
    # acc_filepath = 'class_accuracies/nabirds/group_mohammad_class_accuracy.txt'
    # f = open(f"{acc_filepath}", "a")

    with torch.no_grad():    
        acc, class_acc, embeds, true_labels, predicted_labels = test_epoch(test_loader, model, return_paths=CFG.return_paths)   
        # for k, acc_ in enumerate(class_acc):
        #     f.write(f"{acc_ * 100:.2f}%\n")
        # f.close()
        