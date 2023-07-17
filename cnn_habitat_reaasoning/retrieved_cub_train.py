# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from tqdm import tqdm
import os, random, copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
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
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1468}
    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }
    use_cub_inpaint_test = False

    # retrieved data
    retrieved_data_dir = '/home/tin/projects/reasoning/plain_clip/retrieval_cub_10_images_by_texts_inpaint_unsplash_query/'
    retrieved_cub_df_path = 'cub_retrieved_df.csv'
    write_data_to_df = not os.path.exists(retrieved_cub_df_path)

    # cutmix
    cutmix = False
    cutmix_beta = 1.
    # data params
    n_classes = 1486#200
    test_size = 200
    class_weights = []

    #hyper params
    batch_size = 64
    lr = 1e-3
    image_size = 224
    lr = 0.001
    epochs = 10

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
def Augment(train = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform

# %%    
class Retrieved_CUB_Dataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.df = self.df[~self.df['Path'].str.contains('txt')]
        self.df = self.df[self.df['Mode'] == mode]

        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path, label, mode = self.df.iloc[index].to_list()

        label = int(label)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label #, image_path
    
# %% get data
def compute_class_weights():
    label_folders = os.listdir(CFG.retrieved_data_dir)
    for i, cls in enumerate(label_folders):
        folder_path = f"{CFG.retrieved_data_dir}/{cls}"
        num_samples = len(os.listdir(folder_path))
        CFG.class_weights.append(num_samples)
    CFG.class_weights =[num_sample/sum(CFG.class_weights) for i, num_sample in enumerate(CFG.class_weights)] 
        
compute_class_weights()

train_data_percent = 0.8
test_data_percent = 0.1
valid_data_percent = 0.1

# Get the list of image file paths and their corresponding labels
data = []
label2idx = {}
for i, label in enumerate(os.listdir(CFG.retrieved_data_dir)):
    label2idx[label] = i
for label in os.listdir(CFG.retrieved_data_dir):
    label_folder = os.path.join(CFG.retrieved_data_dir, label)
    if os.path.isdir(label_folder):
        for filename in os.listdir(label_folder):
            image_path = os.path.join(label_folder, filename)
            data.append((image_path, label2idx[label]))

y = [idx for path, idx in data]
y =  np.array(y)
train_data, test_data = train_test_split(data, test_size=test_data_percent, stratify=y, random_state=CFG.seed)
train_data, val_data = train_test_split(train_data, test_size=valid_data_percent, random_state=CFG.seed)

if CFG.write_data_to_df:
    # write to dataframe
    train_df = pd.DataFrame(train_data, columns=["Path", "Label"])
    train_df["Mode"] = ["train" for _ in range(len(train_data))]
    val_df = pd.DataFrame(val_data, columns=["Path", "Label"])
    val_df["Mode"] = ["val" for _ in range(len(val_data))]
    test_df = pd.DataFrame(test_data, columns=["Path", "Label"])
    test_df["Mode"] = ["test" for _ in range(len(test_data))]
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df.to_csv(CFG.retrieved_cub_df_path, index=False)
else:
    df = pd.read_csv(CFG.retrieved_cub_df_path)

# %%
# generate subset based on indices
train_dataset = Retrieved_CUB_Dataset(df, transform=Augment(train=True), mode='train')

if CFG.use_cub_inpaint_test:
    test_dataset = ImageFolder('/home/tin/datasets/cub/CUB_inpaint_all_test/',transform=Augment(train=False))
    val_dataset = test_dataset
else:
    test_dataset = Retrieved_CUB_Dataset(df, transform=Augment(train=False), mode='test')
    val_dataset = Retrieved_CUB_Dataset(df, transform=Augment(train=False), mode='val')

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

 # %%

image,label = train_dataset[10]
image.shape, label

# %%
def display_image(image,label):
     plt.imshow(image.permute(1,2,0))

display_image(*train_dataset[5])
# %% model
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
    best_model = None
    for epoch in range(n_epoch):
        model.train()
        train_loss = training_epoch(trainloader, model)
        print(f'Epoch {epoch}/{n_epoch}, Train Loss: {train_loss}')

        with torch.no_grad():
            model.eval()
            valid_loss, valid_acc = validation_epoch(validloader, model)
            print(f'Epoch {epoch}/{n_epoch}, Valid Loss: {valid_loss}, Valid Acc: {valid_acc*100}%')
            # save model
            if best_valid_acc < valid_acc:
                print('Saving...')
                best_valid_acc = valid_acc
                best_model = model
                torch.save(best_model.state_dict(), f"{CFG.save_folder}/{epoch}_{valid_acc:.3f}_{CFG.dataset}_{CFG.model_name}_cutmix_{CFG.cutmix}_use_cub_inpaint_test_{CFG.use_cub_inpaint_test}.pth")
    return best_model

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

def test(testloader, model):
    accs, losses = [], []
    model.eval()

    for (images, labels) in tqdm(testloader):
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)
        with torch.no_grad():
            out = model(images)                   
        
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)

        losses.append(loss.item())
        accs.append(acc)
    
    return np.mean(losses), np.mean(accs)
# %%
best_model = train(train_loader, val_loader, classification_model, n_epoch = CFG.epochs)

# %% testing
loss, acc = test(test_loader, best_model)
print(f"Test Loss: {loss}, Test Acc: {acc}")