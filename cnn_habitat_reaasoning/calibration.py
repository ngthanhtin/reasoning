"""
TODO: It's always good to put a docstring in scripts (single file executables run from the terminal).
They go here, at the top of the file.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from calibration.utils import load_model_and_buffer

from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck
    
from tqdm import tqdm
from transfg.transfg_vit import VisionTransformer, CONFIGS


class MultiTaskModel_3(nn.Module):
    def __init__(self, num_classes_task=200):
        super(MultiTaskModel_3, self).__init__()

        self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth', map_location=torch.device('cpu'))
        self.backbone1.load_state_dict(my_model_state_dict, strict=True)

        # Freeze backbone (for training only)
        for param in list(self.backbone1.parameters())[:-2]:
            param.requires_grad = False

        self.layer_norm = nn.LayerNorm(num_classes_task)
    def forward(self, x):
        return self.backbone1(x)

# class MultiTaskModel_3(nn.Module):
#     def __init__(self, num_classes_task=555):
#         super(MultiTaskModel_3, self).__init__()

#         self.backbone1 = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
#         my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth', map_location=torch.device('cpu'))
#         self.backbone1.load_state_dict(my_model_state_dict, strict=True)

#         # Freeze backbone (for training only)
#         for param in list(self.backbone1.parameters())[:-2]:
#             param.requires_grad = False

#         self.branch = nn.Sequential(
#             nn.Linear(200, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, num_classes_task)
#         )
    
#     def forward(self, x):
#         features = self.backbone1(x)
#         features = torch.relu(features)
#         features = self.branch(features)
        
#         return features

def Augment(img_size, img_extend_size, train = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(img_extend_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.2, value='random')
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_extend_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, return_paths=False):
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
    
plt.rcParams.update({'font.size': 25})


def correct_and_confidence(model, loader, device, with_energy=True):
    """
    return the correctness and confidences of predictions
    :param model: (obj) model
    :param loader: (iter) train or test loader
    :param with_energy: (bool) use energy or not
    :return: (arr) array of shape (examples, 2) of correct outputs and confidences
    """

    with torch.no_grad():
        model.eval()
        confidences = []
        corrects = []
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if with_energy:
                logits = model.classify(inputs)
            else:
                logits = model(inputs)

            _, predicted = torch.max(logits.data, 1)
            softmaxed_logits = nn.Softmax(dim=1)(logits)
            confidence, _ = torch.max(softmaxed_logits.data, 1)
            confidence = confidence.float().cpu().numpy()
            confidences.extend(confidence)
            correct = (predicted == labels).float().cpu().numpy()
            corrects.extend(correct)

    return np.array(sorted(list(zip(corrects, confidences)), key=lambda x: x[1]))


def calibration_buckets(zipped_corr_conf):
    """
    return calibration buckets
    :param zipped_corr_conf: (arr)
    :return:
            (list) bucket boundaries
            (list) averaged accuracy of examples within each bucket
            (list) averaged confidence of examples within each bucket
            (list) number of examples in each bucket
    """

    thresholds = np.linspace(0, 1, 21)
    corrects = zipped_corr_conf[:, 0]
    confidences = zipped_corr_conf[:, 1]

    buckets = [(thresholds[i], thresholds[i + 1]) for i in range(len(thresholds) - 1)]
    bucket_accs = []
    bucket_confs = []
    bucket_totals = []

    for bucket in buckets:
        total = 0
        correct = 0
        conf = 0
        for i in range(len(confidences)):
            if confidences[i] > bucket[0] and confidences[i] < bucket[1]:
                total += 1
                correct += corrects[i]
                conf += confidences[i]
        if total != 0:
            bucket_acc = correct / total
            bucket_conf = conf / total
        else:
            bucket_acc = 0.
            bucket_conf = 0.
        bucket_accs.append(bucket_acc)
        bucket_confs.append(bucket_conf)
        bucket_totals.append(total)

    return buckets, bucket_accs, bucket_confs, bucket_totals


def expected_calibration_error(data_length, bucket_accs, bucket_confs, bucket_totals):
    """
    compute expected calibration error (ECE)
    :param data_length: (int) number of examples
    :param bucket_accs: (list) averaged accuracy in each bucket
    :param bucket_confs: (list) averaged confidence in each bucket
    :param bucket_totals: (list) number of examples in each bucket
    :return: (float) ECE
    """
    # TODO: If the docstring is longer than the function logic, usually the docstring should be shorter or it shouldn't
    # be a function.
    # Remember that the point of a function is to be an abstraction: something that hides complexity.
    # If it's easier to read the code than understand the docstring, we haven't hidden any complexity.
    # (Often this has to do with the number of arguments.  More args tends to be harder to understand.)

    normalization = (1 / data_length) * np.array(bucket_totals)
    ece = np.abs(np.array(bucket_accs) - np.array(bucket_confs))

    ece = np.dot(normalization, ece)

    return ece

def main(load_dir_sup, sup_type='same', dataset='cub', model_type='mohammad', device='cuda:0', save_inter_calibration=True, save_graph=True):

    bs = 8 if model_type == 'transfg' else 512
    num_classes = 200 if dataset == 'cub' else 555
    img_extend_size = 600 if model_type == 'transfg' else 256
    img_size = 448 if model_type == 'transfg' else 224

    if dataset == 'cub':
        test_data_dir = "/home/tin/datasets/cub/CUB/test/"   
    if dataset == 'nabirds':
        test_data_dir = "/home/tin/datasets/nabirds/test/"

    test_data = ImageFolderWithPaths(root=test_data_dir, transform=Augment(img_size, img_extend_size, train=False))
    testloader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=4)
    
    # Analysis
    # define model
    if model_type == 'cnn':
        supervised_architecture = MultiTaskModel_3(num_classes_task=num_classes)
        model_sup = load_model_and_buffer(load_dir_sup, supervised_architecture, device, with_energy=False)
    if model_type == 'transfg':
        # ViT from TransFG
        config = CONFIGS["ViT-B_16"]
        model_sup = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes, smoothing_value=0.0)
        # model_sup.load_from(np.load("transfg/ViT-B_16.npz"))
        my_model_state_dict = torch.load(load_dir_sup, map_location=torch.device('cpu'))
        model_sup.load_state_dict(my_model_state_dict, strict=True)
        model_sup = model_sup.to(device)
    if model_type == 'mohammad':
        supervised_architecture = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
        my_model_state_dict = torch.load('/home/tin/projects/reasoning/cnn_habitat_reaasoning/visual_correspondence_XAI/ResNet50/CUB_iNaturalist_17/Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth', map_location=torch.device('cpu'))
        supervised_architecture.load_state_dict(my_model_state_dict, strict=True)
        
        # Freeze backbone (for training only)
        for param in list(supervised_architecture.parameters())[:-2]:
            param.requires_grad = False

        model_sup = supervised_architecture.to(device)

    print("computing calibrations ...")
    zipped_corr_conf_sup = correct_and_confidence(model_sup, testloader, device, with_energy=False)

    if save_inter_calibration:
        print("saving intermediate calibrations ...")
        with open(f"./calibration/{dataset}/zipped_corr_conf_supervised_{model_type}-{sup_type}.npy", "wb") as f:
            np.save(f, zipped_corr_conf_sup)
        zipped_corr_conf_sup = np.load(f"./calibration/{dataset}/zipped_corr_conf_supervised_{model_type}-{sup_type}.npy")

    buckets, bucket_accs, bucket_confs, bucket_totals = calibration_buckets(zipped_corr_conf_sup)

    ece_sup = expected_calibration_error(len(testloader.dataset), bucket_accs, bucket_confs, bucket_totals)
    print(f"ece_sup: {ece_sup}")

    ticklabels = [round(i, 1) for i in np.linspace(0, 1, 6)][:-1]

    if save_graph:
        fig = plt.figure(figsize=(20, 10), facecolor="white")

        ax = fig.add_subplot(111)
        ax.bar(np.arange(20), height=bucket_accs) 
        ax.set_xticks(np.arange(0, 20, 4))
        ax.set_xticklabels(ticklabels)
        ax.set_ylim(0, 1)
        x = np.linspace(*ax.get_xlim())
        y = np.linspace(*ax.get_ylim())
        ax.plot(x, y, linestyle='dashed', color='red')
        ax.set_xlabel("bucket")
        ax.set_ylabel("bucket accuracy")

        save_path = f'./calibration/{dataset}/{model_type}/'
        os.makedirs(f'./calibration/{dataset}/', exist_ok=True)
        os.makedirs(f'./calibration/{dataset}/{model_type}/', exist_ok=True)

        
        ax.set_title(f"Ordinary Supervised ({dataset}-{model_type}-{sup_type}): {ece_sup:.4f}", pad=20)
        
        fig.savefig(f"{save_path}/{sup_type}_calibration_plots.png")
    
def test_95_conf_interval(path1, path2):
    zipped_corr_conf_sup1 = np.load(path1)
    zipped_corr_conf_sup2 = np.load(path2)
    corrects1, confidences1 = zipped_corr_conf_sup1[:, 0], zipped_corr_conf_sup1[:, 1]
    corrects2, confidences2 = zipped_corr_conf_sup2[:, 0], zipped_corr_conf_sup2[:, 1]

    from scipy import stats
    mean_set1 = np.mean(confidences1)
    std_dev_set1 = np.std(confidences1, ddof=1)  # Use ddof=1 for sample standard deviation

    mean_set2 = np.mean(confidences2)
    std_dev_set2 = np.std(confidences2, ddof=1)

    # Calculate the confidence intervals for each set
    confidence_level = 0.95
    conf_interval_set1 = stats.norm.interval(confidence_level, loc=mean_set1, scale=std_dev_set1)
    conf_interval_set2 = stats.norm.interval(confidence_level, loc=mean_set2, scale=std_dev_set2)

    print(conf_interval_set1)
    print(conf_interval_set2)
    # Calculate the overlap between the confidence intervals
    overlap = max(0, min(conf_interval_set1[1], conf_interval_set2[1]) - max(conf_interval_set1[0], conf_interval_set2[0]))

    print(f"Confidence interval overlap: {overlap:.2f}")


if __name__ == "__main__":

    #save images
    concat_images = False
    if concat_images:
        from PIL import Image
        # Load the images
        image1 = Image.open('/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/transfg/finetune_calibration_plots.png')
        image2 = Image.open('/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/transfg/same_calibration_plots.png')
        image3 = Image.open('/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/transfg/irrelevant_calibration_plots.png')
        image4 = Image.open('/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/transfg/mix_calibration_plots.png')

        desired_width = image1.width*2
        desired_height = image1.height*2
        # Resize images (if needed) to have the same dimensions
        image1 = image1.resize((desired_width, desired_height))
        image2 = image2.resize((desired_width, desired_height))
        image3 = image3.resize((desired_width, desired_height))
        image4 = image4.resize((desired_width, desired_height))

        # Create a new image with two rows
        combined_image = Image.new('RGB', (desired_width * 2, desired_height * 2))

        # Paste images into the new image
        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (desired_width, 0))
        combined_image.paste(image3, (0, desired_height))
        combined_image.paste(image4, (desired_width, desired_height))
        
        combined_image.save('/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/transfg/combined.png')

        exit()

    test_conf_interval = False
    if test_conf_interval:
        zip1 = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/zipped_corr_conf_supervised_transfg-finetune.npy"
        zip2 = "/home/tin/projects/reasoning/cnn_habitat_reaasoning/calibration/nabirds/zipped_corr_conf_supervised_transfg-irrelevant.npy"
        test_95_conf_interval(zip1, zip2)
        exit()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # cnn cub
    load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/SAME_cub_single_mohammad_08_16_2023-00:32:08/18-0.866-cutmix_False.pth', 'same')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/60_BIRD_ORIG_IRRELEVANT_cub_single_mohammad_08_20_2023-23:34:13/12-0.858-cutmix_False.pth', 'irrelevant')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/MIX_cub_single_mohammad_08_16_2023-00:38:49/19-0.866-cutmix_False.pth', 'mix')

    # transfg cub
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/FINETUNE_cub_single_transfg_08_15_2023-10:37:00/32-0.891-cutmix_False.pth', 'finetune')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/SAME_cub_single_transfg_08_16_2023-00:47:04/45-0.893-cutmix_False.pth', 'same')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/60_BIRD_ORIG_IRRELEVANT_cub_single_transfg_08_20_2023-23:49:17/40-0.888-cutmix_False.pth', 'irrelevant')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/transfg/MIX_cub_single_transfg_08_16_2023-00:48:56/21-0.892-cutmix_False.pth', 'mix')

    # cnn nabirds
    # load_dir_sup = ("/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/FINETUNE_nabirds_single_mohammad_08_14_2023-18:27:21/17-0.802-cutmix_False.pth", 'finetune')
    # load_dir_sup = ("/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/SAME_nabirds_single_mohammad_08_15_2023-00:04:31/18-0.806-cutmix_False.pth", 'same')
    # load_dir_sup = ("/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/MIX_nabirds_single_mohammad_08_15_2023-00:10:47/18-0.807-cutmix_False.pth", 'mix')
    # load_dir_sup = ("/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/mohammad/60_BIRD_ORIG_IRRELEVANT_nabirds_single_mohammad_08_21_2023-01:21:45/19-0.792-cutmix_False.pth", 'irrelevant')
    # transfg nabirds
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/FINETUNE_nabirds_single_transfg_08_17_2023-07:56:43/49-0.884-cutmix_False.pth', 'finetune')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/SAME_nabirds_single_transfg_08_16_2023-01:08:53/31-0.886-cutmix_False.pth', 'same')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/60_BIRD_ORIG_IRRELEVANT_nabirds_single_transfg_08_21_2023-01:22:50/23-0.877-cutmix_False.pth', 'irrelevant')
    # load_dir_sup = ('/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/MIX_nabirds_single_transfg_08_16_2023-14:29:12/48-0.888-cutmix_False.pth', 'mix')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_type = 'cnn'
    dataset = 'cub'

    if model_type == 'mohammad':
        sup_type='mohammad'
        load_dir_sup = ('none', sup_type)

    print(load_dir_sup)

    main(
        load_dir_sup=load_dir_sup[0],
        sup_type= load_dir_sup[1],
        dataset=dataset,
        model_type=model_type,
        device=device,
        save_inter_calibration = False,
        save_graph=False
    )