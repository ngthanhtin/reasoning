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
from albumentations.pytorch import ToTensorV2

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
    dataset = 'cub'
    model_name = 'resnet101' #resnet50, resnet101, efficientnet_b6, densenet121, tf_efficientnetv2_b0
    pretrained = True
    use_inat_pretrained = False
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    # cutmix
    cutmix = True
    cutmix_beta = 1.
    # data params
    n_classes = 200
    test_size = 200

    #hyper params
    batch_size = 64
    lr = 1e-3
    image_size = 224
    lr = 0.001
    epochs = 20

    # explaination
    explaination = False
    

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
train_dataset = ImageFolder('/home/tin/datasets/cub/CUB_inpaint_all_train/',transform=Transforms(Augment('train')))
test_dataset = ImageFolder('/home/tin/datasets/cub/CUB_inpaint_all_test/',transform=Transforms(Augment('test')))
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
if not CFG.use_inat_pretrained:
    classification_model = timm.create_model(
                CFG.model_name,
                pretrained=CFG.pretrained,
                num_classes=CFG.n_classes,
                in_chans=3,
            ).to(CFG.device)
else:
    classification_model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4]).to(CFG.device)
    my_model_state_dict = torch.load('./visual-correspondence-XAI/ResNet-50/CUB-iNaturalist/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
    classification_model.load_state_dict(my_model_state_dict, strict=True)
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
            print(f'Epoch {epoch}/{n_epoch}, Valid Loss: {train_loss}, Valid Acc: {valid_acc*100}')
            # save model
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), f"./{epoch}_{CFG.dataset}_{CFG.model_name}_{valid_acc:.3f}.pth")
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
        for (images, labels) in trainloader:
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
model = train(train_loader, test_loader, classification_model, n_epoch = CFG.epochs)
# %%
if CFG.explaination:
    #What is CaptumInterpretation
    from random import randint
    from matplotlib.colors import LinearSegmentedColormap
    from captum.attr import IntegratedGradients,NoiseTunnel,GradientShap,Occlusion
    from captum.attr import visualization as viz
    from captum.insights import AttributionVisualizer, Batch
    from captum.insights.attr_vis.features import ImageFeature

    from fastai.vision.all import *

    class CaptumInterpretation():
        "Captum Interpretation for Resnet"
        def __init__(self,learn,cmap_name='viridis',colors=None,N=256,methods=('original_image','heat_map'),
                    signs=("all", "positive"),outlier_perc=1):
            if colors is None: colors = [(0, '#ffffff'),(0.25, '#000000'),(1, '#000000')]
            store_attr()
            self.dls,self.model = learn.dls,self.learn.model
            self.supported_metrics=['IG','NT','Occl']

        def get_baseline_img(self, img_tensor,baseline_type):
            baseline_img=None
            if baseline_type=='zeros': baseline_img= img_tensor*0
            if baseline_type=='uniform': baseline_img= torch.rand(img_tensor.shape)
            if baseline_type=='gauss':
                baseline_img= (torch.rand(img_tensor.shape).to(self.dls.device)+img_tensor)/2
            return baseline_img.to(self.dls.device)

        def visualize(self,inp,metric='IG',n_steps=1000,baseline_type='zeros',nt_type='smoothgrad', strides=(3,4,4), sliding_window_shapes=(3,15,15)):
            if metric not in self.supported_metrics:
                raise Exception(f"Metric {metric} is not supported. Currently {self.supported_metrics} are only supported")
            tls = L([TfmdLists(inp, t) for t in L(ifnone(self.dls.tfms,[None]))])
            inp_data=list(zip(*(tls[0],tls[1])))[0]
            enc_data,dec_data=self._get_enc_dec_data(inp_data)
            attributions=self._get_attributions(enc_data,metric,n_steps,nt_type,baseline_type,strides,sliding_window_shapes)
            self._viz(attributions,dec_data,metric)

        def _viz(self,attributions,dec_data,metric):
            default_cmap = LinearSegmentedColormap.from_list(self.cmap_name,self.colors, N=self.N)
            _ = viz.visualize_image_attr_multiple(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(dec_data[0].numpy(), (1,2,0)),
                                                methods=self.methods,
                                                cmap=default_cmap,
                                                show_colorbar=True,
                                                signs=self.signs,
                                                outlier_perc=self.outlier_perc, titles=[f'Original Image - ({dec_data[1]})', metric])



        def _get_enc_dec_data(self,inp_data):
            dec_data=self.dls.after_item(inp_data)
            enc_data=self.dls.after_batch(to_device(self.dls.before_batch(dec_data),self.dls.device))
            return(enc_data,dec_data)

        def _get_attributions(self,enc_data,metric,n_steps,nt_type,baseline_type,strides,sliding_window_shapes):
            # Get Baseline
            baseline=self.get_baseline_img(enc_data[0],baseline_type)
            supported_metrics ={}
            if metric == 'IG':
                self._int_grads = self._int_grads if hasattr(self,'_int_grads') else IntegratedGradients(self.model)
                return self._int_grads.attribute(enc_data[0],baseline, target=enc_data[1], n_steps=200)
            elif metric == 'NT':
                self._int_grads = self._int_grads if hasattr(self,'_int_grads') else IntegratedGradients(self.model)
                self._noise_tunnel= self._noise_tunnel if hasattr(self,'_noise_tunnel') else NoiseTunnel(self._int_grads)
                return self._noise_tunnel.attribute(enc_data[0].to(self.dls.device), n_samples=1, nt_type=nt_type, target=enc_data[1])
            elif metric == 'Occl':
                self._occlusion = self._occlusion if hasattr(self,'_occlusion') else Occlusion(self.model)
                return self._occlusion.attribute(enc_data[0].to(self.dls.device),
                                        strides = strides,
                                        target=enc_data[1],
                                        sliding_window_shapes=sliding_window_shapes,
                                        baselines=baseline)
    captum=CaptumInterpretation(model, colors=['green','red','yellow'])
    # %%s
    path = '../input/bird-species-classification-220-categories/Train'
    def get_image_files(path):
        filenames = os.listdir(path)
        filepaths = [f"{path}/{fname}" for fname in filenames if 'txt' not in fname]
        return filepaths
    #%%
    fnames = get_image_files(path)
    idx=randint(0,len(fnames))
    captum.visualize(fnames[idx])

    # %%
    captum.visualize(fnames[idx],metric='Occl',baseline_type='gauss')
    # %%
    captum.visualize(fnames[idx],metric='IG',baseline_type='uniform')

