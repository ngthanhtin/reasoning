# %%
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

import os, random
import numpy as np

import timm
import albumentations as A

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
    model_name = 'resnet50'
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    lr = 0.01
    n_classes = 6

set_seed(CFG.seed)

# %% Augmentation
def Augment(mode):
    if mode == 'train':
        train_aug_list = [transforms.Resize(256), transforms.CenterCrop(224), 
                               transforms.ToTensor(), 
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        return transforms.Compose(train_aug_list)
    else:
        valid_test_aug_list = [transforms.Resize(256), transforms.CenterCrop(224), 
                               transforms.ToTensor(), 
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        return transforms.Compose(valid_test_aug_list)
    
    
# %% Dataset
test_data_dir = './intel_dataset/seg_test/seg_test/'
train_data_dir ='./intel_dataset/seg_train/seg_train/'
 
# %%
train_dataset = ImageFolder(train_data_dir, transform=Augment('train'))

test_dataset = ImageFolder(test_data_dir,transform=Augment('valid'))

 # %%

image,label = train_dataset[0]
image.shape, label

# %%
print("Follwing classes are there : \n",train_dataset.classes)

def display_image(image,label):
     plt.imshow(image.permute(1,2,0))

display_image(*train_dataset[0])

# %%
test_size = 2000
train_size = len(train_dataset) - test_size

train_data,test_data = random_split(train_dataset,[train_size,test_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(test_data)}")

# %%
train_loader = DataLoader(train_data,CFG.batch_size,shuffle=True,num_workers = 4, pin_memory = True)
valid_loader = DataLoader(test_data,CFG.batch_size,num_workers = 4, pin_memory = True)

# %% model

classification_model = timm.create_model(
            CFG.model_name,
            pretrained=True,
            num_classes=CFG.n_classes,
            in_chans=3,
        ).to(CFG.device)

# %%
num_epochs = 10
lr = 0.001
optimizer = torch.optim.Adam(classification_model.parameters(), lr=lr)

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

def train(trainloader, validloader, model, n_epoch=10, fold=0):
    best_valid_acc = 0.0
    for epoch in range(n_epoch):
        model.train()
        train_loss = training_epoch(trainloader, model)
        print(f'Epoch {epoch}/{n_epoch}, Train Loss: {train_loss}')

        with torch.no_grad():
            model.eval()
            valid_loss, valid_acc = validation_epoch(validloader, model)
            print(f'Epoch {epoch}/{n_epoch}, Valid Loss: {train_loss}, Valid Acc: {valid_acc}')
            # save model
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), f"./{epoch}_{valid_acc}.pth")
    return model

def training_epoch(trainloader, model):
        losses = []
        for (images, labels) in trainloader:
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
    
    for (images, labels) in validloader:
        images = images.to(CFG.device)
        labels = labels.to(CFG.device)

        out = model(images)                   
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)

        losses.append(loss.item())
        accs.append(acc)
    
    return np.mean(losses), np.mean(accs)

# %%
model = train(train_loader, valid_loader, classification_model, n_epoch = 12, fold=0)
# %%
