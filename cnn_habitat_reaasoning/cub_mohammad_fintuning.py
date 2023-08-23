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
    model_name = 'transfg' #mohammad, vit, transfg
    use_cont_loss = True
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

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
    orig_train_img_folder = 'temp_gen_data/CUB_aug_irrelevant_with_orig_birds_train_60/' # 'CUB_irrelevant_augmix_train_small', 'CUB_augmix_train_small/', 'CUB_aug_train_4_small'
    #CUB/test, CUB_inpaint_all_test (onlybackground), CUB_no_bg_test, CUB_random_test, CUB_bb_on_birds_test, CUB_big_bb_on_birds_test, CUB_nobirds_test (blackout-birds)
    orig_test_img_folder = 'CUB/test/'
    orig_test_img_folder = 'CUB_inpaint_all_test/'
    orig_test_img_folder = 'CUB_no_bg_test/'
    orig_test_img_folder = 'CUB_random_test/'
    orig_test_img_folder = 'CUB_bb_on_birds_test/'
    orig_test_img_folder = 'CUB_big_bb_on_birds_test/'
    # test with inat
    orig_test_img_folder = '../overlapping_cub_inat/'

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
        orig_train_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_train_img_folder}"
        orig_test_data_dir = f"{CFG.dataset2path[dataset]}/{CFG.orig_test_img_folder}"

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

    for inputs, bird_labels, paths in tqdm(testloader):
        inputs = inputs.to(CFG.device)
        bird_labels = bird_labels.to(CFG.device)
        bird_outputs = model(inputs)

        _, preds = torch.max(bird_outputs, 1)
        probs, _ = torch.max(F.softmax(bird_outputs, dim=1), 1)
        running_corrects += torch.sum(preds == bird_labels.data)


    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    print('-' * 10)
    print('Acc: {:.4f}'.format(100*epoch_acc))

    return 100*epoch_acc


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
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/MIX_cub_single_mohammad_08_16_2023-00:38:49/19-0.866-cutmix_False.pth' # mix

    # transfg
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/FINETUNE_cub_single_transfg_08_15_2023-10:37:00/32-0.891-cutmix_False.pth' #finetune transfg only
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/60_BIRD_ORIG_IRRELEVANT_cub_single_transfg_08_20_2023-23:49:17/40-0.888-cutmix_False.pth' # irrelevant
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/SAME_cub_single_transfg_08_16_2023-00:47:04/45-0.893-cutmix_False.pth' # aug_same
    model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/MIX_cub_single_transfg_08_16_2023-00:48:56/21-0.892-cutmix_False.pth' # augmix
    print(model_path)
    print(CFG.orig_test_img_folder)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # write result to file
    acc_filepath = model_path.replace(model_path.split('/')[-1], 'accuracy.txt')
    f = open(f"{acc_filepath}", "a")
    f.write(f"{model_path}, {CFG.orig_test_img_folder}\n")
    if not CFG.test_tta:
        with torch.no_grad():    
            acc = test_epoch(test_loader, model, return_paths=CFG.return_paths)   
            f.write(f"{acc:.4f}\n")
            f.close()
    else:
        from image_retrieval import find_image_by_image, load_model, load_features_and_paths

        # Test TTA (Orig and Blackbox)
        class Dataset_TTA(Dataset):
            def __init__(self, root1, root2, transform1=None, transform2=None):
                self.root1 = root1
                self.root2 = root2
                self.transform1 = transform1
                self.transform2 = transform2

                label_folders = os.listdir(self.root1)
                self.img_pair_files = []
                self.targets = []

                # init CLIP model
                retrieval_model, preprocess, tokenizer = load_model('ViT-L/14', device=CFG.device)
                image_features, image_paths = load_features_and_paths()

                for folder in tqdm(label_folders):
                    label = int(folder.split('.')[0])
                    img_files = os.listdir(f"{self.root1}/{folder}")
                    # img_files2 = os.listdir(f"{self.root2}/{folder}")

                    for file in img_files:
                        path1 = f"{self.root1}/{folder}/{file}"
                        returned_image_paths = find_image_by_image(retrieval_model, preprocess, image_features, image_paths, path1, n=10, device=CFG.device)
                        # path2 = f"{self.root2}/{folder}/{file}"
                        path2 = returned_image_paths[0]
                        path3 = returned_image_paths[1]
                        path4 = returned_image_paths[2]
                        
                        # file2 = random.choice(img_files2)
                        # file3 = random.choice(img_files2)
                        # file4 = random.choice(img_files2)
                        # path2 = f"{self.root2}/{folder}/{file2}"
                        # path3 = f"{self.root2}/{folder}/{file3}"
                        # path4 = f"{self.root2}/{folder}/{file4}"

                        #
                        # self.img_pair_files.append([path1, path2, path3, path4])
                        self.img_pair_files.append(returned_image_paths)
                        self.targets.append(label)

            def __len__(self):        
                return len(self.img_pair_files)
                
            def __getitem__(self, index):        
                label = self.targets[index] - 1
                orig_img  = Image.open(self.img_pair_files[index][0]).convert("RGB")
                second_img = Image.open(self.img_pair_files[index][1]).convert("RGB")

                # blackbox_img1 = Image.open(self.img_pair_files[index][1]).convert("RGB")
                blackbox_img2 = Image.open(self.img_pair_files[index][2]).convert("RGB")
                blackbox_img3 = Image.open(self.img_pair_files[index][3]).convert("RGB")
                blackbox_img4 = Image.open(self.img_pair_files[index][4]).convert("RGB")
                blackbox_img5 = Image.open(self.img_pair_files[index][5]).convert("RGB")
                blackbox_img6 = Image.open(self.img_pair_files[index][6]).convert("RGB")
                blackbox_img7 = Image.open(self.img_pair_files[index][7]).convert("RGB")
                blackbox_img8 = Image.open(self.img_pair_files[index][8]).convert("RGB")
                blackbox_img9 = Image.open(self.img_pair_files[index][9]).convert("RGB")
                

                if self.transform1 is not None:
                    orig_img = torch.tensor(self.transform1(orig_img))
                if self.transform2 is not None:
                    second_img = torch.tensor(self.transform2(second_img))
                    # blackbox_img1 = torch.tensor(self.transform(blackbox_img1))
                    blackbox_img2 = torch.tensor(self.transform2(blackbox_img2))
                    blackbox_img3 = torch.tensor(self.transform2(blackbox_img3))
                    blackbox_img4 = torch.tensor(self.transform2(blackbox_img4))
                    blackbox_img5 = torch.tensor(self.transform2(blackbox_img5))
                    blackbox_img6 = torch.tensor(self.transform2(blackbox_img6))
                    blackbox_img7 = torch.tensor(self.transform2(blackbox_img7))
                    blackbox_img8 = torch.tensor(self.transform2(blackbox_img8))
                    blackbox_img9 = torch.tensor(self.transform2(blackbox_img9))
                    
                    imgs = torch.cat((orig_img, second_img, blackbox_img2, blackbox_img3, blackbox_img4, blackbox_img5, blackbox_img6, blackbox_img7, blackbox_img8, blackbox_img9), 0)
                    # imgs = orig_img
                
                # imgs = torch.cat((orig_img, second_img), 0)

                return imgs, label

        # random.seed(47)
        tta_transform = transforms.Compose([
            transforms.Resize(CFG.image_expand_size),
            transforms.CenterCrop(CFG.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.RandomErasing(p=0.2, value='random')
        ])
        # /home/tin/datasets/cub/CUB_inpaint_all_test/, CUB_bb_on_birds_test, CUB_big_bb_on_birds_test
        tta_dataset = Dataset_TTA(root1='/home/tin/datasets/cub/CUB/test/', root2='/home/tin/datasets/cub/CUB/test/', transform1=Augment(train=False), transform2=tta_transform)
        tta_test_loader = DataLoader(tta_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=False, pin_memory=False)

        index2folderpath = {}
        folder_path2imgfiles = {}
        train_cub_dir = '/home/tin/datasets/cub/CUB/train/'
        label_folders = os.listdir(train_cub_dir)
        for folder in label_folders:
            folder_idx = int(folder.split(".")[0]) - 1
            
            folder_path = f"{train_cub_dir}/{folder}"
            index2folderpath[folder_idx] = folder_path

            img_files = os.listdir(folder_path)
            img_paths = [f"{train_cub_dir}/{folder}/{f}" for f in img_files]
            folder_path2imgfiles[folder_path] = img_paths

        def tta_test_epoch(testloader, model):
            model.eval()
            full_paths = []
            running_corrects = 0

            val_transform=Augment(train=False)

            for inputs, bird_labels in tqdm(testloader):
                bird_labels = bird_labels.to(CFG.device)

                orig_inputs = inputs[:, :3, :, :].to(CFG.device)
                second_inputs = inputs[:, 3:6, :, :].to(CFG.device)
                # bb_inputs1 = inputs[:, 3:6, :, :].to(CFG.device)
                bb_inputs2 = inputs[:, 6:9, :, :].to(CFG.device)
                bb_inputs3 = inputs[:, 9:12, :, :].to(CFG.device)
                bb_inputs4 = inputs[:, 12:15, :, :].to(CFG.device)
                bb_inputs5 = inputs[:, 15:18, :, :].to(CFG.device)
                bb_inputs6 = inputs[:, 18:21, :, :].to(CFG.device)
                bb_inputs7 = inputs[:, 21:24, :, :].to(CFG.device)
                bb_inputs8 = inputs[:, 24:27, :, :].to(CFG.device)
                bb_inputs9 = inputs[:, 27:, :, :].to(CFG.device)
                
                
                orig_outputs = model(orig_inputs)
                second_outputs = model(second_inputs)
                # bb_outputs1 = model(bb_inputs1)
                bb_outputs2 = model(bb_inputs2)
                bb_outputs3 = model(bb_inputs3)
                bb_outputs4 = model(bb_inputs4)
                bb_outputs5 = model(bb_inputs5)
                bb_outputs6 = model(bb_inputs6)
                bb_outputs7 = model(bb_inputs7)
                bb_outputs8 = model(bb_inputs8)
                bb_outputs9 = model(bb_inputs9)
                

                outputs = F.softmax(orig_outputs) + F.softmax(second_outputs) + F.softmax(bb_outputs2) + F.softmax(bb_outputs3) #+ \
                #+ F.softmax(bb_outputs4)# + #F.softmax(bb_outputs5) + F.softmax(bb_outputs6) + F.softmax(bb_outputs7) + \
                #F.softmax(bb_outputs8) + F.softmax(bb_outputs9)
                # Get top-k indices
                # indices = torch.argmax(outputs, 1)
                # values, indices = torch.topk(outputs, k=2, dim=1)
            
                # batch_added_inputs1 = []
                # batch_added_inputs2 = []
                # batch_added_inputs3 = []
                # for i, top_idx in enumerate(indices):
                #     top1_idx, top2_idx = top_idx[0].item(), top_idx[1].item()
                #     query_folder_path = index2folderpath[top1_idx]
                #     img_files = folder_path2imgfiles[query_folder_path]
                #     added_file1 = random.choice(img_files)
                #     if (values[i,0] - values[i,1]).item() >= 7.:
                #         top2_idx = top1_idx
                #     query_folder_path = index2folderpath[top2_idx]
                #     img_files = folder_path2imgfiles[query_folder_path]
                #     added_file2 = random.choice(img_files)
                #     # added_file3 = random.choice(img_files)

                #     added_img1  = Image.open(added_file1).convert("RGB")
                #     batch_added_inputs1.append(torch.tensor(val_transform(added_img1)))
                #     added_img2  = Image.open(added_file2).convert("RGB")
                #     batch_added_inputs2.append(torch.tensor(val_transform(added_img2)))
                    # added_img3  = Image.open(added_file3).convert("RGB")
                    # batch_added_inputs3.append(torch.tensor(val_transform(added_img3)))

                # batch_added_inputs1 = torch.stack(batch_added_inputs1, dim=0)
                # batch_added_inputs1 = batch_added_inputs1.to(CFG.device)
                # batch_added_outputs1 = model(batch_added_inputs1)

                # batch_added_inputs2 = torch.stack(batch_added_inputs2, dim=0)
                # batch_added_inputs2 = batch_added_inputs2.to(CFG.device)
                # batch_added_outputs2 = model(batch_added_inputs2)

                # # batch_added_inputs3 = torch.stack(batch_added_inputs3, dim=0)
                # # batch_added_inputs3 = batch_added_inputs3.to(CFG.device)
                # # batch_added_outputs3 = model(batch_added_inputs3)

                # outputs = F.softmax(outputs) + F.softmax(batch_added_outputs1) + F.softmax(batch_added_outputs2) #+ F.softmax(batch_added_outputs3)

                
                _, preds = torch.max(outputs, 1)
                probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
                running_corrects += torch.sum(preds == bird_labels.data)


            epoch_acc = running_corrects.double() / len(testloader.dataset)

            print('-' * 10)
            print('Acc: {:.4f}'.format(100*epoch_acc))

            return 100*epoch_acc

        with torch.no_grad():    
            acc = tta_test_epoch(tta_test_loader, model)   
            f.write(f"{acc:.4f}\n")
            f.close()