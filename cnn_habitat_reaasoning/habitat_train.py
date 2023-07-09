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
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # data params
    n_classes = 200
    test_size = 100

    #hyper params
    batch_size = 64
    lr = 0.01
    image_size = 224
    lr = 0.001
    epochs = 12

    

set_seed(CFG.seed)

# %% Augmentation
def Augment(mode):
    if mode == 'train':
        train_aug_list = [transforms.Resize(256), 
                        transforms.CenterCrop(224),
                        transforms.RandomRotation(10), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        # transforms.RandomErasing(p=0.5),
                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        return transforms.Compose(train_aug_list)
    else:
        valid_test_aug_list = [transforms.Resize(256), 
                               transforms.CenterCrop(224), 
                               transforms.ToTensor(), 
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        return transforms.Compose(valid_test_aug_list)
    
    
# %% Dataset
train_data_dir ='/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_text_2/'
test_data_dir ='/home/tin/projects/reasoning/inpainting/cub_inpaint/'
 
 #  %%
from PIL import Image
import torch
from torch.utils.data import Dataset

class HabitatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
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
            self.file_list.extend([os.path.join(class_folder, img_file) for img_file in image_files])
            self.labels.extend([label] * len(image_files))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.file_list[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        return image, label
# %%
train_dataset = ImageFolder(train_data_dir, transform=Augment('train'))
test_dataset = ImageFolder(test_data_dir,transform=Augment('valid'))

 # %%

image,label = train_dataset[10]
image.shape, label

# %%
def display_image(image,label):
     plt.imshow(image.permute(1,2,0))

display_image(*train_dataset[5])

# %%
test_size = CFG.test_size
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
CFG.lr = 1e-4
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
model = train(train_loader, valid_loader, classification_model, n_epoch = CFG.epochs)
# %%
