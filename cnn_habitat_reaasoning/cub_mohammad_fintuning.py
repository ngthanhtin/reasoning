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
    seed = 42
    dataset = 'cub'
    model_name = 'mohammad' #mohammad, vit, transfg
    use_cont_loss = True
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1486}
    bird_num_classes = dataset2num_classes[dataset]
    habitat_num_classes = dataset2num_classes[dataset]
    # train, test data paths
    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }
    orig_train_img_folder = 'CUB/train/'#'temp_gen_data/CUB_aug_irrelevant_with_orig_birds_train_60/' # 'CUB_irrelevant_augmix_train_small', 'CUB_augmix_train_small/', 'CUB_aug_train_4_small'
    #CUB/test, CUB_inpaint_all_test (onlybackground), CUB_no_bg_test, CUB_random_test, CUB_bb_on_birds_test, CUB_big_bb_on_birds_test, CUB_nobirds_test (blackout-birds)
    orig_test_img_folder = 'CUB/test/'
    # orig_test_img_folder = 'CUB_inpaint_all_test/'
    # orig_test_img_folder = 'CUB_no_bg_test/'
    # orig_test_img_folder = 'CUB_random_test/'
    # orig_test_img_folder = 'CUB_bb_on_birds_test/'
    # orig_test_img_folder = 'CUB_big_bb_on_birds_test/'
    # test with inat
    # orig_test_img_folder = '../overlapping_cub_inat/'

    # test with fly-nonfly birds
    # orig_test_img_folder = '../flybird_cub_test/'
    # orig_test_img_folder = '../non_flybird_cub_test/'

    # cutmix
    cutmix = False

    #hyper params
    lr = 1e-5 if model_name in {'vit', 'transfg'} else 1e-4
    image_size = 224 if model_name in {'mohammad', 'vit'} else 448
    image_expand_size = 256 if model_name in {'mohammad', 'vit'} else 600
    epochs = 50 if model_name in {'vit', 'transfg'} else 20

    # train or test
    train = False
    return_paths = not train
    batch_size = 64
    if model_name == 'transfg':
        batch_size = 8
    else:
        batch_size = 64 if train else 512

    test_tta = False
    # inat21
    inat21_df_path = 'inat21_onlybirds.csv'
    write_inat_to_df = not os.path.exists(inat21_df_path)

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
    if dataset in ['cub', 'nabirds']:
        orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_train_img_folder}"
        orig_test_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_test_img_folder}"

        train_data = ImageFolderWithPaths(root=orig_train_data_dir, transform=Augment(train=True)) #, num_images_per_class=0)
        test_data = ImageFolderWithPaths(root=orig_test_data_dir, transform=Augment(train=False)) #, num_images_per_class=3)
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
from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
    
class MultiTaskModel_3(nn.Module):
    def __init__(self, num_classes_task1=CFG.bird_num_classes, num_classes_task2=CFG.habitat_num_classes):
        super(MultiTaskModel_3, self).__init__()

        self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth', map_location=torch.device('cpu'))
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

def test_epoch(testloader, model, return_paths=False):
    model.eval()
    full_paths = []
    running_corrects = 0

    class_corrects = [0] * CFG.bird_num_classes
    class_counts = [0] * CFG.bird_num_classes

    confusion_matrix = np.zeros((CFG.bird_num_classes, CFG.bird_num_classes), dtype=int)

    for inputs, bird_labels, paths in tqdm(testloader):
        inputs = inputs.to(CFG.device)
        bird_labels = bird_labels.to(CFG.device)
        bird_outputs = model(inputs)

        _, preds = torch.max(bird_outputs, 1)
        probs, _ = torch.max(F.softmax(bird_outputs, dim=1), 1)
        running_corrects += torch.sum(preds == bird_labels.data)

        for i in range(len(bird_labels)):
            true_label = bird_labels[i]
            predicted_label = preds[i]
            class_corrects[bird_labels[i]] += int(preds[i] == bird_labels[i])
            class_counts[bird_labels[i]] += 1
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
if CFG.model_name == 'mohammad':
    model = MultiTaskModel_3(num_classes_task1=200)
if CFG.model_name == 'vit':
    model = ViTBase16(n_classes=200, pretrained=True)
if CFG.model_name == 'transfg':
    # ViT from TransFG
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, 448, zero_head=True, num_classes=200, smoothing_value=0.0)
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
    # mohammad
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/60_BIRD_ORIG_IRRELEVANT_cub_single_mohammad_08_20_2023-23:34:13/12-0.858-cutmix_False.pth' # irrelevant with orig birds
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/SAME_cub_single_mohammad_08_16_2023-00:32:08/18-0.866-cutmix_False.pth' # same
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/MIX_cub_single_mohammad_08_16_2023-00:38:49/19-0.866-cutmix_False.pth' # mix

    model_path = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/cub_single_mohammad_11_06_2023-00:47:07/11-0.864-cutmix_False.pth" #mohammad finetune

    # transfg
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/FINETUNE_cub_single_transfg_08_15_2023-10:37:00/32-0.891-cutmix_False.pth' #finetune transfg only
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/60_BIRD_ORIG_IRRELEVANT_cub_single_transfg_08_20_2023-23:49:17/40-0.888-cutmix_False.pth' # irrelevant
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/SAME_cub_single_transfg_08_16_2023-00:47:04/45-0.893-cutmix_False.pth' # aug_same
    # model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/MIX_cub_single_transfg_08_16_2023-00:48:56/21-0.892-cutmix_False.pth' # augmix
    print(model_path)
    print(CFG.orig_test_img_folder)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # write result to file
    acc_filepath = model_path.replace(model_path.split('/')[-1], 'accuracy.txt')
    top_misclassified_filepath = model_path.replace(model_path.split('/')[-1], 'augsame_missclassification.txt')
    # f = open(f"{acc_filepath}", "a")
    f = open(top_misclassified_filepath, 'w')
    f.write(f"{model_path}, {CFG.orig_test_img_folder}\n")
    if not CFG.test_tta:
        with torch.no_grad():    
            acc, class_acc, confusion_matrix = test_epoch(test_loader, model, return_paths=CFG.return_paths)   
            top_misclassified_counts = 0
            for i in range(200):
                top_misclassified_indices, counts = get_top_misclassified_classes(confusion_matrix, i, top_k=1)
                top_misclassified_counts+=counts
                f.write(f"{idx_to_class[i]}*{idx_to_class[top_misclassified_indices[0]]}*{counts[0]}\n")
            # f.write(f"{acc:.4f}\n")
            f.close()

            # save class accuracy
            # import csv
            # sup_type = 'mix'
            # csv_file_path = f"/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/{sup_type}_class_accuracies.csv"
            # with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            #     csv_writer = csv.writer(csv_file)
            #     csv_writer.writerow(["Class", "Accuracy"])
            #     for class_idx, accuracy in enumerate(class_acc):
            #         csv_writer.writerow([idx_to_class[class_idx], accuracy])