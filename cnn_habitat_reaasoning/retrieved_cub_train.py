# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

from tqdm import tqdm
import os, random, copy
import numpy as np
from datetime import datetime

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import clip

from visual_correspondence_XAI.ResNet50.CUB_iNaturalist_17.FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
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

# %% config
class CFG:
    seed = 42
    dataset = 'cub' # cub
    model_name = 'resnet101' #resnet50, resnet101, efficientnet_b6, densenet121, tf_efficientnetv2_b0
    pretrained = True
    use_inat_pretrained = False
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1468}
    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }

    # cutmix
    cutmix = False
    cutmix_beta = 1.
    # data params
    n_classes = 1486#200
    test_size = 200

    #hyper params
    batch_size = 128
    lr = 1e-3
    image_size = 224
    lr = 0.001
    epochs = 20

    # explaination
    explaination = False
    
     # save folder
    save_folder    = f'./results/retrieved_cub_{model_name}_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

set_seed(CFG.seed)

# %% use transforms with albumentations
class Transforms:
    def __init__(self, album_transforms):
        self.transforms = album_transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']
# %% Augmentation
def Augment(mode):
    if mode == 'train':
        return A.Compose([A.RandomResizedCrop(224,224),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.OneOf([ #
                    A.GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
                    A.GaussianBlur(blur_limit=(3,7), p=0.5),], p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.3, 
                                 brightness_by_max=True,p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, 
                    sat_shift_limit=0.2, 
                    val_shift_limit=0.2, 
                    p=0.5),
                # A.CoarseDropout(p=0.5),
                # A.Cutout(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        
    else:
        return A.Compose([A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
    
# %% Dataset
# data_dir ='/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts/'
# data_dir ='/home/tin/projects/reasoning/inpainting/cub_inpaint/'
 
 #  %%
from PIL import Image
import torch
from torch.utils.data import Dataset

class HabitatDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.root = root
        self.augment = Augment(mode)

        self.file_list = []
        self.labels = []
        self._load_files()

    def _load_files(self):

        class_folders = sorted(os.listdir(self.root))
        for label, class_folder in enumerate(class_folders):
            class_path = os.path.join(self.root, class_folder)
            if not os.path.isdir(class_path):
                continue
            image_files = os.listdir(class_path)
            self.file_list.extend([os.path.join(class_folder, img_file) for img_file in image_files if 'txt' not in img_file])
            self.labels.extend([label] * len(image_files))
        
        self.train_list = self.file_list[:int(len(self.file_list)*0.8)] 
        self.train_labels = self.labels[:int(len(self.labels)*0.8)]
        self.test_list = self.file_list[int(len(self.file_list)*0.8):] 
        self.test_labels = self.labels[int(len(self.labels)*0.8):]

        if self.mode == 'train':
            self.file_list = self.train_list
            self.labels = self.train_labels
        else:
            self.file_list = self.test_list
            self.labels = self.test_labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.file_list[index])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        image = self.augment(image)

        return image, label
# %%
# train_dataset = HabitatDataset('/home/tin/datasets/cub/CUB_no_bg_train/', mode='train')
# test_dataset = HabitatDataset('/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts', mode='test')
# train_dataset = ImageFolder('/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts_inpaint_unsplash_query/',transform=Transforms(Augment('train')))
# cub
train_dataset = ImageFolder('/home/tin/datasets/cub/CUB_inpaint_all_train/',transform=Transforms(Augment('train')))
test_dataset = ImageFolder('/home/tin/datasets/cub/CUB_inpaint_all_test/',transform=Transforms(Augment('test')))
#inat
all_dataset = ImageFolder('/home/tin/datasets/inaturalist2021_onlybird/inat21_inpaint_all/', transform=Transforms(Augment('train')))
train_data_len = int(len(all_dataset)*0.78)
valid_data_len = int((len(all_dataset) - train_data_len)/2)
test_data_len = int(len(all_dataset) - train_data_len - valid_data_len)
train_dataset, val_dataset, test_dataset = random_split(all_dataset, [train_data_len, valid_data_len, test_data_len])
 # %%

image,label = train_dataset[10]
image.shape, label

# %%
def display_image(image,label):
     plt.imshow(image.permute(1,2,0))

display_image(*train_dataset[5])

# %%
# test_size = CFG.test_size
# train_size = len(train_dataset) - test_size

# train_data,test_data = random_split(train_dataset,[train_size,test_size])
# print(f"Length of Train Data : {len(train_data)}")
# print(f"Length of Validation Data : {len(test_data)}")

# %%
train_loader = DataLoader(train_dataset,CFG.batch_size,shuffle=True,num_workers = 4, pin_memory = True)
test_loader = DataLoader(test_dataset,CFG.batch_size,num_workers = 4, pin_memory = True)

# %% model
print(CFG.model_name)
if CFG.model_name in ['resnet101', 'resnet50']:
    classification_model = timm.create_model(
                CFG.model_name,
                pretrained=CFG.pretrained,
                num_classes=CFG.n_classes,
                in_chans=3,
            ).to(CFG.device)
# else:
#     classification_model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4]).to(CFG.device)
#     my_model_state_dict = torch.load('./visual-correspondence-XAI/ResNet-50/CUB-iNaturalist/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
#     classification_model.load_state_dict(my_model_state_dict, strict=True)

elif CFG.model_name == 'clip':
    clip_model, transform = clip.load('ViT-L/14', device=CFG.device)
    
    visual_encoder = clip_model.visual
    visual_encoder.fc = nn.Identity()

    
    for param in visual_encoder.parameters():
        param.requires_grad = False
    
    classification_model = nn.Sequential(
                visual_encoder,
                nn.ReLU(),
                nn.Linear(visual_encoder.output_dim, CFG.n_classes)
                ).to(CFG.device).to(torch.float32)
# %%
optimizer = torch.optim.Adam(classification_model.parameters(), lr=CFG.lr)

# %%
def show_batch_images(dataloader):
    for images,labels in dataloader:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

show_batch_images(train_loader)

# %%
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
# %%

def train(trainloader, validloader, model, n_epoch=10):
    best_valid_acc = 0.0
    for epoch in range(n_epoch):
        model.train()
        train_loss = training_epoch(trainloader, model)
        print(f'Epoch {epoch}/{n_epoch}, Train Loss: {train_loss}')

        with torch.no_grad():
            model.eval()
            valid_loss, valid_acc = validation_epoch(validloader, model)
            print(f'Epoch {epoch}/{n_epoch}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc*100}')
            # save model
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), f"./{epoch}_{CFG.dataset}_{CFG.cutmix}_{CFG.model_name}_{valid_acc:.3f}.pth")
    return model

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

def cutmix_same_class(images, labels, alpha):
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

    return torch.tensor(mixed_images), torch.tensor(mixed_labels)

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

show_batch_cutmix_images(test_loader)
# %%
def training_epoch(trainloader, model):
        losses = []
        for (images, labels) in tqdm(trainloader):
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)
            if CFG.cutmix and random.random() > 0.4:
                # lam = np.random.beta(CFG.cutmix_beta, CFG.cutmix_beta)
                # rand_index = torch.randperm(images.size()[0])
                # bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)    
                # images[:, bbx1:bbx2, bby1:bby2, :] = images[rand_index, bbx1:bbx2, bby1:bby2, :]
                images, labels = cutmix_same_class(images, labels, CFG.cutmix_beta)
                images = images.to(CFG.device)
                labels = labels.to(CFG.device)

            out = model(images)
            loss = F.cross_entropy(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)
    
def validation_epoch(validloader, model):
    accs, losses = [], []
    
    for (images, labels) in tqdm(validloader):
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)

        out = model(images)                   
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)

        losses.append(loss.item())
        accs.append(acc)
    
    return np.mean(losses), np.mean(accs)

# %%
model = train(train_loader, test_loader, classification_model, n_epoch = CFG.epochs)

