import os, json, copy
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.transforms import Compose

from PIL import Image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, selected_folders, transform=None):
        self.data_dir = data_dir
        self.selected_folders = selected_folders
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for folder_name in self.selected_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            class_index = self.selected_folders.index(folder_name)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    data.append((file_path, class_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')
        # img_root = os.path.join(root, 'CUB_fixed/test/')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)
        path = self.imgs[index][0]

        real_sample = sample.copy()
        
        to_tensor = transforms.ToTensor()
        real_sample = to_tensor(real_sample)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target, path
    
    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2: 
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict
                
                

from PIL import Image
import torchvision.transforms as transforms

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    

import os
import json
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from nabirds_horn import load_bounding_box_annotations, load_part_annotations, load_part_names, load_class_names, \
     load_image_labels, load_image_paths, load_image_sizes, load_hierarchy, load_photographers, load_train_test_split

import random

class PartImageNetDataset(Dataset):
    def __init__(self, root_dir: str, description_dir: str, transform: Compose = None, train: bool = True, n_pixel: int = 224) -> None:
        super().__init__()
        if not train:
            self.data_dir = f"{root_dir}/images/test_folders/"
            self.transform = transform
            self.totensor = transforms.ToTensor()
            self.loader=datasets.folder.default_loader
            self.n_pixel = n_pixel

            self.description_dir = description_dir
            self.synset2name_dir = "/home/tin/projects/reasoning/plain_clip/descriptors/part_imagenet/pi_synset2name.json"
            
        self.load_meta()

    def load_meta(self):
        # load synset mapping
        with open(self.synset2name_dir, "r") as f:
            synset_2_classname = json.load(f)

        with open(self.description_dir, 'r') as f:
            classname_2_descriptions = json.load(f)
        
        classname_2_idx = {} # load from description files
        for i, (classname, desc) in enumerate(classname_2_descriptions.items()):
            classname_2_idx[classname] = i

        self.samples = []
        self.targets = []
        class_folders = os.listdir(self.data_dir)
        for class_fol in class_folders: # synset
            if class_fol not in synset_2_classname:
                continue
            image_names = os.listdir(f"{self.data_dir}/{class_fol}")
            for name in image_names:
                self.samples.append(f"{self.data_dir}/{class_fol}/{name}")
                classname = synset_2_classname[class_fol]
                self.targets.append(classname_2_idx[classname])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx], self.targets[idx]

        sample = self.loader(image_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, image_path




class NABirdsDataset(Dataset):
    allowed_keys = ['crop', 'box_dir', 'return_path', 'trivial_aug', 'ops', 'high_res', 'n_pixel', 'return_mask', 'all_data', 'zeroshot_split']
    part_mapping = {'beak': [0], 'crown': [1], 'nape': [3], 'eyes': [3, 4], 'belly': [5], 'breast': [6], 'back': [7], 'tail': [8], 'wings': [9, 10]}

    def __init__(self, root_dir: str, transform: Compose = None, train: bool = True, subset_class_names: list = [], zeroshot_split: bool = False, **kwargs):

        self.root_dir = root_dir
        self.subset_class_names = subset_class_names
        self.zeroshot_split = zeroshot_split

        self.transform = transform
        self.train = train
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.allowed_keys})
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)

        self._load_meta()

        self.loader = default_loader
        self.totensor = transforms.ToTensor()

    def _load_meta(self):

        # ================== Load meta data ==================
        # Loading code adopt form NABirds sample script with minor modification
        self.image_paths = load_image_paths(self.root_dir, path_prefix='images')
        self.image_sizes = load_image_sizes(self.root_dir)
        self.image_bboxes = load_bounding_box_annotations(self.root_dir)
        self.image_parts = load_part_annotations(self.root_dir)
        self.image_class_labels = load_image_labels(self.root_dir)

        # Load in the class data
        self.class_names = load_class_names(self.root_dir)
        self.class_hierarchy = load_hierarchy(self.root_dir)

        # Load in the part data
        self.dataset_parts = load_part_names(self.root_dir)
        self.part_ids = list(self.dataset_parts.keys())
        self.part_ids = sorted(self.part_ids)

        # Load in the train / test split
        self.train_images, self.test_images = load_train_test_split(self.root_dir)

        # ===================================================
        # replace part names 'bill' with 'beak'
        # (this is the parts from the dataset, not to confused with part_name used for training, e.g., self.part_names)
        self.dataset_parts = {0: 'beak', 1: 'crown', 2: 'nape', 3: 'left eye', 4: 'right eye', 5: 'belly',
                              6: 'breast', 7: 'back', 8: 'tail', 9: 'left wing', 10: 'right wing'}

        all_samples = self.train_images + self.test_images
        all_classes = [int(self.image_class_labels[image_id]) for image_id in all_samples]
        unique_classes = set(all_classes)
        unique_class_names = [self.class_names[str(class_id)] for class_id in unique_classes]
        
        # there are 555 classes in total, re-index them from 0 to 554
        self.class2idx = {v: k for k, v in enumerate(unique_class_names)}
        self.idx2class = dict(enumerate(unique_class_names))
        self.nabirds_idx2class_idx = dict(zip(unique_classes, self.idx2class))
        self.class_idx2nabirds_idx = dict(zip(self.idx2class, unique_classes))

        self.subset_class_labels = []
        for sub_cls in self.subset_class_names:
            for k, v in self.idx2class.items():
                if v == sub_cls:
                    self.subset_class_labels.append(int(k))
                    break
                
        # training samples
        self.target_classes = unique_classes
        self.zs_class2class_id = None
        self.class_id2zs_class = None

        if self.zeroshot_split:
            # choose 25% of the classes for testing (with seed 77)
            random.seed(77)     # Fixed random seed
            test_classes = set(random.sample(unique_classes, int(len(unique_classes) * 0.25)))
            train_classes = set(unique_classes) - test_classes
            all_imgs = self.train_images + self.test_images

            self.target_classes = train_classes if self.train else test_classes
            # reindex the classes to 0, 1, 2, ...
            self.zs_class2class_id = dict(enumerate(self.target_classes))
            self.class_id2zs_class = {v: k for k, v in self.zs_class2class_id.items()}
            image_ids = [image_id for image_id in all_imgs if int(self.image_class_labels[image_id]) in self.target_classes]
        elif hasattr(self, 'all_data') and self.all_data:
            image_ids = all_samples
        else:
            image_ids = self.train_images if self.train else self.test_images

        self.samples = []
        self.targets = []
        self.sample_parts = {}
        self.image_boxes = {}

        for image_id in image_ids:
            image_path = self.image_paths[image_id]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            class_label = self.nabirds_idx2class_idx[int(self.image_class_labels[image_id])]
            
            if self.class_id2zs_class is not None:
                class_label = self.class_id2zs_class[class_label]

            if class_label not in self.subset_class_labels:
                continue

            self.targets.append(class_label)
            self.samples.append([os.path.join(self.root_dir, image_path), class_label])
            self.sample_parts[image_name] = self.image_parts[image_id]
            self.image_boxes[image_name] = self.image_bboxes[image_id]

        self.classes = list(self.class2idx.keys())

        sorted_subset_class_labels = sorted(self.subset_class_labels)
        
        for sample in self.samples:
            sample[1] = sorted_subset_class_labels.index(sample[1])
    
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    # overwrite __getitem__
    def __getitem__(self, idx):
        image_path, target = self.samples[idx]

        sample = self.loader(image_path)
        sample_size = torch.tensor(sample.size[::-1])  # (h, w)

        if self.transform is not None:
            sample = self.transform(sample)

        if hasattr(self, 'return_path') and self.return_path:
            return sample, target, image_path

        return sample, target, image_path
    



class INaturalistDataset(Dataset):
    allowed_keys = ['crop', 'box_dir', 'return_path', 'trivial_aug', 'ops', 'high_res', 'return_mask']
    def __init__(self, root_dir:str, box_dir: str = None, transform: Compose = None, subset_class_names: list = [], n_pixel: int = 336, train: bool = True, original: bool = False, **kwargs):
        self.root_dir = root_dir
        self.transform = transform
        self.n_pixel = n_pixel
        self.box_dir = box_dir
        self.train = train
        self.subset_class_names = subset_class_names

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        with open(os.path.join(root_dir, 'bird_annotations.json'), 'r') as f:
            self.annotation_meta = json.load(f)
        with open(os.path.join(root_dir, 'bird_classes.json'), 'r') as f:
            self.class_meta = json.load(f)
        # easy mapping for class id, class name and corresponding folder name
    
        self.idx2class = {int(id): name for id, name in self.class_meta['name'].items() if name in self.subset_class_names} # if use subset
        self.class2idx = {v: k for k, v in self.idx2class.items()}

        self.cls_id2folder_name = {int(id): name for id, name in self.class_meta['image_dir_name'].items() if name in self.subset_class_names} # if use subset
        self.folder_name2cls_id = {v: k for k, v in self.cls_id2folder_name.items()}
        with open(os.path.join(root_dir, 'bird_images.json'), 'r') as f:
            self.image_meta = json.load(f)

        self.images = self.image_meta['file_name']
        self.images = {int(k): v for k, v in self.images.items()}
        self.samples = []
        self.targets = []
        for k, v in self.images.items():
            if self.annotation_meta['category_id'][str(k)] - 1 in self.idx2class.keys(): # if use subset
                self.samples.append(os.path.join(root_dir, v))
                self.targets.append(self.annotation_meta['category_id'][str(k)] - 1) # -1 to make it zero-indexed
        
        ## ---- Re-index---- # if use subset
        self.classes = list(self.class2idx.keys())
        self.subset_class_labels = list(self.idx2class.keys())
        sorted_subset_class_labels = sorted(self.subset_class_labels)
        
        for i, sample in enumerate(self.samples):
            self.targets[i] = sorted_subset_class_labels.index(self.targets[i])
        # ------------------

        self.samples = tuple(zip(self.samples, self.targets))
        self.targets = tuple(self.targets)
        self.loader = default_loader
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.n_pixel, self.n_pixel), interpolation=Image.BICUBIC, antialias=True)

        if hasattr(self, 'trivial_aug') and self.trivial_aug:
            # self.tri_aug = TrivialAugmentCustom(operations=self.ops)
            self.tri_aug = None
        else:
            self.tri_aug = None

        if not original and self.box_dir:
            self.filter_data()

    def filter_data(self,):
        # filter out the images based on the box outputs.
        box_files = os.listdir(self.box_dir)
        image_ids = [os.path.splitext(f)[0] for f in box_files]
        keep = [os.path.splitext(os.path.basename(image_id))[0] in image_ids for image_id in list(self.images.values())]
        self.samples = tuple(self.samples[i] for i in range(len(self.samples)) if keep[i])
        self.targets = tuple(self.targets[i] for i in range(len(self.targets)) if keep[i])

    def __len__(self):
        return len(self.samples)
    # overwrite __getitem__

    def __getitem__(self, idx):
        image_path, class_id = self.samples[idx]
        image_path = image_path.replace('train_mini', 'bird_train')

        sample = self.loader(image_path)
        sample_size = torch.tensor(sample.size[::-1]) # (h, w)
        hight, width = sample_size
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        img_tensor = self.totensor(sample)
        # if not hasattr(self, 'return_path') or not self.return_path:
        #     # repeat samples
        #     boxes = torch.load(os.path.join(self.box_dir, f'{image_id}.pth'))["boxes_info"]
        #     sample = torch.cat([self.resize(img_tensor).unsqueeze(0)] * len(boxes))
        #     masks = torch.zeros((len(boxes), self.n_pixel, self.n_pixel), dtype=torch.uint8)
        #     if hasattr(self, 'crop') or hasattr(self, 'return_mask'):
        #         crop_samples = []
        #         for idx, (part_name, box) in enumerate(boxes.items()):
        #             # boxes = {'Part name0': [x0, y0, x1, y1], 'Part name1': [x0, y0, x1, y1], ...}
        #             if self.crop:
        #                 crop_samples.append(self.resize(img_tensor[:, box[1]:box[3], box[0]:box[2]].unsqueeze(0)))
        #             if self.return_mask:
        #                 # scale box from image_size (h, w) to (n_pixel, n_pixel)
        #                 scale_factor = torch.tensor([self.n_pixel / width, self.n_pixel / hight, self.n_pixel / width, self.n_pixel / hight])
        #                 box = (torch.tensor(box) * scale_factor).int()
        #                 masks[idx, box[1]:box[3], box[0]:box[2]] = 1
        #         crop_sample = torch.cat(crop_samples)
        #     if len(crop_samples) == 0:
        #         crop_samples = torch.zeros_like(sample)
        # else:
            # sample = sample
        if self.tri_aug is not None and self.train:
            sample = self.tri_aug(sample)
        if self.transform:
            sample = self.transform(sample)

        if hasattr(self, 'return_path') and self.return_path:
            return sample, class_id, image_path, sample_size

        return sample, class_id, image_path #, masks, crop_sample