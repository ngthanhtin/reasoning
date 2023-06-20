import random

from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, VisionDataset, ImageNet, Places365, EuroSAT, DTD, Food101, OxfordIIITPet
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from torch.utils.data import Subset

from .cub import CUB
from configs import *


class DatasetWrapper(VisionDataset):
    def __init__(self, dataset_name, transform, distortion=None, samples_per_class: int = None, random_seed: int = 42):
        self.transform = transform
        self.dataset_name = dataset_name

        # Used for ImageNet-C only
        results = distortion.split(":") if distortion is not None else [None, None]
        self.distortion_name, self.distortion_severity = results[0], results[1]

        self.dataset, self.n_classes = self.load_dataset(dataset_name, samples_per_class, random_seed)

    def load_dataset(self, dataset_name, samples_per_class, random_seed: int = 42):
        random.seed(random_seed)

        if dataset_name == DS_IMAGENET:
            # dataset = ImageNet(IMAGENET_DIR, split='val', transform=self.transform)
            dataset = ImageFolder(IMAGENET_DIR, transform=self.transform)
        elif dataset_name == DS_IMAGENET_V2:
            dataset = ImageNetV2(location=IMAGENET_V2_DIR, transform=self.transform)
        elif dataset_name == DS_IMAGENET_A:
            dataset = ImageFolder(root=IMAGENET_A_DIR, transform=self.transform)
        elif dataset_name == DS_IMAGENET_C:
            dataset = ImageFolder(root=f"{IMAGENET_C_DIR}/{self.distortion_name}/{self.distortion_severity}/", transform=self.transform)
        elif dataset_name == DS_PLACES365:
            dataset = Places365(PLACES365_DIR, split='val', transform=self.transform)
        elif dataset_name == DS_CUB:
            dataset = CUB(CUB_DIR, train=False, transform=self.transform)

        n_classes = len(dataset.classes) if dataset_name != DS_IMAGENET_V2 else 1000

        # Using Subset
        if samples_per_class is not None and samples_per_class > 0:
            class_counts = self.get_class_count(dataset_name, dataset)

            selected_indices = []
            total_samples_processed = 0
            for label, max_count in class_counts.items():
                selected_indices.extend(sorted(list(random.sample(list(range(total_samples_processed, total_samples_processed + max_count)), k=min(samples_per_class, max_count)))))
                total_samples_processed += max_count
            dataset = Subset(dataset, selected_indices)

        return dataset, n_classes

    def __getitem__(self, idx):
        sample, target = self.dataset.__getitem__(idx)
        path, image_id = "", ""

        if self.dataset_name in [DS_IMAGENET, DS_IMAGENET_A, DS_IMAGENET_C, DS_PLACES365, DS_CUB]:
            path, label = self.dataset.dataset.imgs[self.dataset.indices[idx]] if isinstance(self.dataset, Subset) else self.dataset.imgs[idx]
            image_id = ".".join(path.split("/")[-1].split(".")[:-1])
        elif self.dataset_name == DS_IMAGENET_V2:
            posix_path = self.dataset.dataset.fnames[self.dataset.indices[idx]] if isinstance(self.dataset, Subset) else self.dataset.fnames[idx]
            path, image_id = str(posix_path), posix_path.stem

        return sample, target, path, image_id

    def __len__(self):
        return len(self.dataset)

    def get_class_count(self, dataset_name, dataset):
        class_counts = {}

        if dataset_name in [DS_IMAGENET, DS_IMAGENET_A, DS_IMAGENET_C, DS_PLACES365, DS_CUB]:
            for path, label in dataset.imgs:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        elif dataset_name == DS_IMAGENET_V2:
            for posix_path in dataset.fnames:
                label = posix_path.parts[-2]
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        return class_counts
