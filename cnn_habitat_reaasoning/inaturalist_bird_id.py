# %%
import numpy as np
import pandas as pd

# %%
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision.utils import make_grid
from torchvision import models

# %%
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import create_transform


# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import os
import sys
from tqdm import tqdm
import random
import time
import copy

# %% config
class CFG:
    seed = 42
    dataset = 'inat21' # cub, nabirds, inat21
    model_name = 'resnet101' #resnet50, resnet101, efficientnet_b6, densenet121, tf_efficientnetv2_b0
    pretrained = True
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    # data params
    dataset2num_classes = {'cub': 200, 'nabirds': 555, 'inat21':1468}
    dataset2path = {
        'cub': '/home/tin/datasets/cub',
        'nabirds': '/home/tin/datasets/nabirds/',
        'inat21': '/home/tin/datasets/inaturalist2021_onlybird/'
    }
    is_inpaint = True

    # cutmix
    cutmix = False
    cutmix_beta = 1.

    #hyper params
    batch_size = 128
    lr = 1e-3
    image_size = 224
    epochs = 15

    # explaination
    explaination = False

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
def get_data_loaders(dataset, batch_size, train = False):
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
        if dataset in ['cub', 'nabirds']:
            if dataset == 'cub':
                img_folder = 'CUB_inpaint_all_train/' if CFG.is_inpaint else 'CUB/train/'
            else:
                img_folder = 'train_inpaint/' if CFG.is_inpaint else 'train/'

            train_data_dir = f"{CFG.dataset2path[dataset]}/{img_folder}"
            train_data = datasets.ImageFolder(train_data_dir, transform=transform)
            train_data_len = len(train_data)
            classes = train_data.classes
        elif dataset == 'inat21':
            data_dir = CFG.dataset2path[dataset] + '/bird_train'
            all_data = datasets.ImageFolder(data_dir, transform=transform)

            train_data_len = int(len(all_data)*0.78) 
            valid_data_len = int((len(all_data) - train_data_len)/2) 
            test_data_len = int(len(all_data) - train_data_len - valid_data_len) 
        
            train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
            classes = all_data.classes
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

        return train_loader, train_data_len, classes
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if dataset in ['cub', 'nabirds']:
            if dataset == 'cub':
                img_folder = 'CUB_inpaint_all_test/' if CFG.is_inpaint else 'CUB/test/'
            else:
                img_folder = 'test_inpaint/' if CFG.is_inpaint else 'test/'

            test_data_dir = f"{CFG.dataset2path[dataset]}/{img_folder}"
            test_data = datasets.ImageFolder(test_data_dir, transform=transform)
            val_data = test_data
            valid_data_len = len(val_data)
            test_data_len = len(test_data)
            classes = test_data.classes
        elif dataset == 'inat21':
            data_dir = CFG.dataset2path[dataset] + '/bird_train'
            all_data = datasets.ImageFolder(data_dir, transform=transform)
            train_data_len = int(len(all_data)*0.78) # 80%
            valid_data_len = int((len(all_data) - train_data_len)/2) # 10%
            test_data_len = int(len(all_data) - train_data_len - valid_data_len) # 10%
            train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
            classes = all_data.classes
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

        return (val_loader, test_loader, valid_data_len, test_data_len, classes)

# %%
(train_loader, train_data_len, classes) = get_data_loaders(CFG.dataset, CFG.batch_size, train=True)
(val_loader, test_loader, valid_data_len, test_data_len, classes) = get_data_loaders(CFG.dataset, CFG.batch_size, train=False)

# %%
dataloaders = {
    "train":train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train":train_data_len,
    "val": valid_data_len
}

# %%
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

# %%
print(train_data_len, test_data_len, valid_data_len)

# %%
def formatText(class_label):
    return " ".join(class_label.split("_")[-2:])
formatText(classes[0])

# %%
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
#     plt.imshow(np.transpose(images[idx], (1, 2, 0)))
#     ax.set_title(formatText(classes[labels[idx]]))
#     plt.show()
# %%
# model = models.efficientnet_b3(pretrained=True)
model_name = 'resnet101' # tf_efficientnetv2_b0
model = timm.create_model(model_name, pretrained=True)
# model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
# %%
for param in model.parameters():
    param.requires_grad = False

if model_name == 'tf_efficientnetv2_b0':
    n_inputs = model.classifier.in_features
    model.classifier = nn.Sequential(
    nn.Linear(n_inputs,2048),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(2048, len(classes)))
    model_params = model.classifier.parameters()
elif model_name in ['resnet50', 'resnet101']:
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, len(classes))
    )
    model_params = model.fc.parameters()
    
model = model.to(CFG.device)

# %%
criterion = LabelSmoothingCrossEntropy()
# criterion = nn.CrossEntropyLoss()
criterion = criterion.to(CFG.device)
optimizer = optim.Adam(model_params, lr=CFG.lr)

# %%
training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}

# %%
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

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
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                if phase == 'train' and CFG.cutmix and random.random() > 0.4:
                    # lam = np.random.beta(CFG.cutmix_beta, CFG.cutmix_beta)
                    # rand_index = torch.randperm(images.size()[0])
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)    
                    # images[:, bbx1:bbx2, bby1:bby2, :] = images[rand_index, bbx1:bbx2, bby1:bby2, :]
                    images, labels = cutmix_same_class(images, labels, CFG.cutmix_beta)
                    images = images.to(CFG.device)
                    labels = labels.to(CFG.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                training_history['accuracy'].append(epoch_acc)
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(epoch_acc)
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=CFG.epochs)

# %%
torch.cuda.empty_cache()

# %%
from tqdm import tqdm
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

model_ft.eval()

for data, target in tqdm(test_loader):
    if torch.cuda.is_available(): 
        data, target = data.to(CFG.device), target.to(CFG.device)
    with torch.no_grad():
        output = model_ft(data)
        loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)    
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    if len(target) == CFG.batch_size:
      for i in range(CFG.batch_size):
          label = target.data[i]
          class_correct[label] += correct[i].item()
          class_total[label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

# for i in range(len(classes)):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             formatText(classes[i]), 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

test_acc = round(100. * np.sum(class_correct) / np.sum(class_total), 3)
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    test_acc,
    np.sum(class_correct), np.sum(class_total)))

# %% save model tradditionally
torch.save(model.state_dict(), f"traditionally_{CFG.dataset}-{test_acc}-{model_name}-inpaint_{CFG.is_inpaint}.pth")
# %%
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model_ft.cpu(), example)
traced_script_module.save(f"{CFG.dataset}-{test_acc}-{model_name}-inpaint_{CFG.is_inpaint}.pth")

# %%



