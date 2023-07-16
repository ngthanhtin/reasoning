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
import seaborn as sns

import os
import sys
from tqdm import tqdm
import random
import time
import copy

# %% config
class CFG:
    seed = 42
    dataset = 'nabirds' # cub, nabirds, inat21
    model_name = 'resnet101' #resnet50, resnet101, efficientnet_b6, densenet121, tf_efficientnetv2_b0
    pretrained = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
    batch_size = 64
    lr = 1e-3
    image_size = 224
    epochs = 15

    # explaination
    explaination = False

    # ensemble
    ensemble = True

    # inat21
    inat21_df_path = 'inat21_onlybirds.csv'
    write_inat_to_df = not os.path.exists(inat21_df_path)

    # %%
    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}

    # focal loss
    fl_alpha = 1.0  # alpha of focal_loss
    fl_gamma = 2.0  # gamma of focal_loss
    class_weights = []

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

class Inat21_Dataset(Dataset):
    def __init__(self, df, transform=None, mode='train', inpaint=False):
        self.df = df
        self.df = self.df[self.df['Mode'] == mode]
        self.mode = mode
        self.transform = transform
        self.inpaint = inpaint

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path, label, mode = self.df.iloc[index].to_list()
        if self.inpaint:
            image_path = image_path.replace("bird_train", "inat21_inpaint_all")

        label = int(label)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
def get_data_loaders(dataset, batch_size):
    """
    Get the train, val, test dataloader
    """
    if dataset in ['cub', 'nabirds']:
        if dataset == 'cub':
            train_img_folder = 'CUB_inpaint_all_train/' if CFG.is_inpaint else 'CUB/train/'
            test_img_folder = 'CUB_inpaint_all_test/' if CFG.is_inpaint else 'CUB/test/'
        else:
            train_img_folder = 'train_inpaint/' if CFG.is_inpaint else 'train/'
            test_img_folder = 'test_inpaint/' if CFG.is_inpaint else 'test/'

        # train data
        train_data_dir = f"{CFG.dataset2path[dataset]}/{train_img_folder}"
        train_data = datasets.ImageFolder(train_data_dir, transform=Augment(train=True))
        train_data_len = len(train_data)

        # val, test data
        test_data_dir = f"{CFG.dataset2path[dataset]}/{test_img_folder}"
        test_data = datasets.ImageFolder(test_data_dir, transform=Augment(train=False))
        val_data = test_data
        valid_data_len = len(val_data)
        test_data_len = len(test_data)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        classes = train_data.classes

    elif dataset == 'inat21':
        if not CFG.is_inpaint:
            data_dir = CFG.dataset2path[dataset] + '/bird_train'
        else:
            data_dir = CFG.dataset2path[dataset] + '/inat21_inpaint_all'

        def compute_class_weights():
            label_folders = os.listdir(data_dir)
            for i, cls in enumerate(label_folders):
                folder_path = f"{data_dir}/{cls}"
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
        for i, label in enumerate(os.listdir(data_dir)):
            label2idx[label] = i
        for label in os.listdir(data_dir):
            label_folder = os.path.join(data_dir, label)
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, filename)
                    data.append((image_path, label2idx[label]))
        
        y = [idx for path, idx in data]
        y =  np.array(y)
        train_data, test_data = train_test_split(data, test_size=test_data_percent, stratify=y, random_state=CFG.seed)
        train_data, val_data = train_test_split(train_data, test_size=valid_data_percent, random_state=CFG.seed)

        if CFG.write_inat_to_df:
            # write to dataframe
            train_df = pd.DataFrame(train_data, columns=["Path", "Label"])
            train_df["Mode"] = ["train" for _ in range(len(train_data))]
            val_df = pd.DataFrame(val_data, columns=["Path", "Label"])
            val_df["Mode"] = ["val" for _ in range(len(val_data))]
            test_df = pd.DataFrame(test_data, columns=["Path", "Label"])
            test_df["Mode"] = ["test" for _ in range(len(test_data))]
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            df.to_csv(CFG.inat21_df_path, index=False)
        else:
            df = pd.read_csv(CFG.inat21_df_path)
        # generate subset based on indices
        train_dataset = Inat21_Dataset(df, transform=Augment(train=True), mode='train', inpaint=CFG.is_inpaint)
        test_dataset = Inat21_Dataset(df, transform=Augment(train=False), mode='test', inpaint=CFG.is_inpaint)
        val_dataset = Inat21_Dataset(df, transform=Augment(train=False), mode='val', inpaint=CFG.is_inpaint)

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

        classes = os.listdir(data_dir)
        train_data_len, valid_data_len, test_data_len = len(train_dataset), len(val_dataset), len(test_dataset)

    return (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes)
# %%
(train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes) = get_data_loaders(CFG.dataset, CFG.batch_size)
# %%
visualize_loader = False
if CFG.dataset == 'inat21' and visualize_loader:
    class_counts = defaultdict(int)

    for _, labels in train_loader:
        for label in labels:
            label = label.item()
            class_counts[label] += 1
    print("Counting finished !!!")

    num_bins = 5  
    num_classes = len(class_counts)
    print("Num classes: ", num_classes)

    bins = defaultdict(int)
    other_bin_count = 0
    bin_id = 1
    for i, (class_label, count) in enumerate(class_counts.items()):
        if i < (num_classes/num_bins)*bin_id:
            bins[bin_id] += count
        else:
            bin_id += 1
    
    for bin_id, count in bins.items():
        print(f"Bin {bin_id}: {count} samples")

    # Plot the reduced class statistics
    plt.bar(bins.keys(), bins.values())
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    plt.title("Reduced Class Statistics")
    plt.show()
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

def get_model():
    # model = models.efficientnet_b3(pretrained=True)
    model_name = CFG.model_name #'resnet101' # tf_efficientnetv2_b0
    model = timm.create_model(model_name, pretrained=True)
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

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
        # n_inputs = model.fc.in_features
        # model.fc = nn.Sequential(
        #     nn.Linear(n_inputs,2048),
        #     nn.SiLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(2048, len(classes))
        # )
        # model_params = model.fc.parameters()
        model = timm.create_model(
                CFG.model_name,
                pretrained=CFG.pretrained,
                num_classes=len(classes),
                in_chans=3,
            ).to(CFG.device)
        model_params = model.parameters()

    return model, model_params

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

# %%
def train(trainloader, validloader, optimizer, criterion, scheduler, model, num_epochs = 10):
    
    best_acc = 0.
    for epoch in range(num_epochs):
        print("")
        model.train()
        train_loss, train_acc = train_epoch(trainloader, model, criterion, optimizer)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}") #, LR: {scheduler.get_lr()}")
        
        with torch.no_grad():    
            valid_loss, valid_acc = evaluate_epoch(validloader, criterion, model)     
            print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")
            # save model
            if best_acc <= valid_acc:
                print("Saving...")
                best_acc = valid_acc
                torch.save(model.state_dict(), f"{CFG.dataset}-{epoch}-{best_acc:.3f}-{CFG.model_name}-inpaint_{CFG.is_inpaint}.pth")
        
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

        if CFG.cutmix and random.random() > 0.4:
            # lam = np.random.beta(CFG.cutmix_beta, CFG.cutmix_beta)
            # rand_index = torch.randperm(images.size()[0])
            # bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)    
            # images[:, bbx1:bbx2, bby1:bby2, :] = images[rand_index, bbx1:bbx2, bby1:bby2, :]
            images, labels = cutmix_same_class(images, labels, CFG.cutmix_beta)
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # statistics
        losses.append(loss.item())
        accs.append((torch.sum(preds == labels.data)/CFG.batch_size).detach().cpu().numpy())
            
    return np.mean(losses), np.mean(accs)

# %%
def evaluate_epoch(validloader, criterion, model):
    model.eval()
    losses = []
    accs = []

    for inputs, labels in tqdm(validloader):
        inputs = inputs.to(CFG.device)

        outputs = model(inputs).detach().cpu() 
        _, preds = torch.max(outputs, 1)
        criterion = criterion.to('cpu')
        loss = criterion(outputs, labels)
        criterion.to(CFG.device)

        # statistics
        losses.append(loss.item())
        accs.append(torch.sum(preds == labels.data)/CFG.batch_size)
            
    return np.mean(losses), np.mean(accs)

# %%
"""
Define Focal-Loss
"""
class FocalLoss(nn.Module):
    """
    The focal loss for fighting against class-imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12  # prevent training from Nan-loss error
        self.cls_weights = torch.tensor([CFG.class_weights],dtype=torch.float, requires_grad=False, device=CFG.device)

    def forward(self, logits, target):
        """
        logits & target should be tensors with shape [batch_size, num_classes]
        """
        probs = torch.sigmoid(logits)
        one_subtract_probs = 1.0 - probs
        # add epsilon
        probs_new = probs + self.epsilon
        one_subtract_probs_new = one_subtract_probs + self.epsilon
        # calculate focal loss
        print(target.shape, probs_new.shape)
        log_pt = target * torch.log(probs_new) + (1.0 - target) * torch.log(one_subtract_probs_new)
        pt = torch.exp(log_pt)
        focal_loss = -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss * self.cls_weights
        return torch.mean(focal_loss)#, probs
# %%
if not CFG.ensemble:
    model, model_params = get_model()
    model.to(CFG.device)

    criterion = LabelSmoothingCrossEntropy()
    # criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(CFG.class_weights).to(CFG.device))
    # criterion = FocalLoss()
    criterion = criterion.to(CFG.device)
    optimizer = optim.Adam(model_params, lr=CFG.lr)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    model_ft = train(train_loader, val_loader, optimizer, criterion, exp_lr_scheduler, model, num_epochs=CFG.epochs)
    with torch.no_grad():    
        test_loss, test_acc = evaluate_epoch(test_loader, criterion, model_ft)   
        print(f"Test Loss: {test_loss}, Valid Acc: {test_acc}")

# %%
if not CFG.ensemble:

    test_loss = 0.0
    class_correct = [0. for _ in range(len(classes))]
    class_total = [0. for _ in range(len(classes))]

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

# %% ensemble
def test_cub_ensemble(id_model, habitat_model, id_loader, habitat_loader, alpha=1):
    id_model.eval()
    habitat_model.eval()
    
    running_loss = 0.0
    running_corrects = 0
  
    predictions = []
    targets = []
    paths = []
    confidence = []

    criterion = nn.CrossEntropyLoss()
    bs = CFG.batch_size

    with torch.inference_mode():
        for _, ((data1, target1), (data2, target2)) in tqdm(enumerate(zip(id_loader, habitat_loader))): # dont shuffle the loaders
            data1   = data1.to(CFG.device)
            data2   = data2.to(CFG.device)
            for tg1 in target1:
                if tg1 not in target2:
                    print('WRONG')
            else:
                target = target1
            target = target.to(CFG.device)
        
            id_feat = id_model(data1)
            habitat_feat = habitat_model(data2)
            outputs = id_feat*alpha + habitat_feat*(1-alpha)
            # outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)
            probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
            running_loss += loss.item() * target.size(0)
            running_corrects += torch.sum(preds == target.data)
            
            predictions.extend(preds.data.cpu().numpy())
            targets.extend(target.data.cpu().numpy())
        
            confidence.extend((probs.data.cpu().numpy()*100).astype(np.int32))

        epoch_loss = running_loss / (len(id_loader)*bs)
        epoch_acc = running_corrects.double() / (len(id_loader)*bs)

    print('-' * 10)
    print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100*epoch_acc))
    
    return predictions, targets, confidence

# %%
if CFG.ensemble:
    # get models
    id_model, params = get_model()
    id_model.load_state_dict(torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/nabirds-11-0.743-resnet101-inpaint_False.pth'))
    id_model.to(CFG.device)
    habitat_model, params = get_model()
    habitat_model.load_state_dict(torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/nabirds-4-0.065-resnet101-inpaint_True.pth'))
    habitat_model.to(CFG.device)
    # get loaders
    CFG.is_inpaint = False
    (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes) = get_data_loaders(CFG.dataset, CFG.batch_size)
    id_test_loader = test_loader
    CFG.is_inpaint = True
    (train_loader, val_loader, test_loader, train_data_len, valid_data_len, test_data_len, classes) = get_data_loaders(CFG.dataset, CFG.batch_size)
    habitat_test_loader = test_loader

    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(alpha)
        cub_test_preds, _, cub_test_confs = test_cub_ensemble(id_model, habitat_model, id_test_loader, habitat_test_loader, alpha=alpha)
# %%
