# %% [markdown]
# # Method 1 - Using iNaturalist-pretrained ResNet-50

# %%
import random
random.seed(43)

# %%
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
# %%
from FeatureExtractors import ResNet_AvgPool_classifier, Bottleneck

# %%
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device

# %%
val_dataset_transform = transforms.Compose(
  [transforms.Resize(256), 
  transforms.CenterCrop(224), 
  transforms.ToTensor(), 
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# %%
class ImageFolderWithPaths(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, num_images_per_class=3):
        super(ImageFolderWithPaths, self).__init__(root, transform, target_transform)
        self.root = root

        if num_images_per_class != 0:
          self.num_images_per_class = num_images_per_class
          self._limit_dataset()

    def _limit_dataset(self):
        new_data = []
        new_targets = []
        for class_idx in range(len(self.classes)):
          class_data = [item for item in self.samples if item[1] == class_idx]
          selected_samples = random.sample(class_data, min(self.num_images_per_class, len(class_data)))
          data, targets = zip(*selected_samples)
          new_data.extend(data)
          new_targets.extend(targets)
        self.samples = list(zip(new_data, new_targets))

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        
        return (img, label ,path)

class ImageFolderWithTwoPaths(ImageFolder):
    def __init__(self, root1, root2, transform=None, target_transform=None):
        super(ImageFolderWithTwoPaths, self).__init__(root1, transform, target_transform)
        self.root2 = root2

    def __getitem__(self, index):
  
        # img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        # path = self.imgs[index][0]
        
        path, label = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        path2 = self.root2 +"/" + path.split("/")[-2] + "/" + path.split("/")[-1]
        img2 = self.loader(path2)

        if self.transform is not None:
            img2 = self.transform(img2)

        return (img, img2, label, path, path2)
# %%
# validation_folder = ImageFolder(root='/home/tin/datasets/cub/CUB/test', transform=val_dataset_transform)
validation_folder = ImageFolderWithPaths(root='/home/tin/datasets/non_flybird_cub_test/', transform=val_dataset_transform, num_images_per_class=3)
val_loader        = DataLoader(validation_folder, batch_size=512, shuffle=False, num_workers=8, pin_memory=False)

# %% [markdown]
# ## iNAT ResNet-50 

# %%
inat_resnet = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
my_model_state_dict = torch.load('./Forzen_Method1-iNaturalist_avgpool_200way1_85.83_Manuscript.pth')
inat_resnet.load_state_dict(my_model_state_dict, strict=True)

# %%
# Dimension of classification head
print(list(inat_resnet.parameters())[-2].shape)

# %%
# Freeze backbone (for training only)
for param in list(inat_resnet.parameters())[:-2]:
  param.requires_grad = False
    
# to CUDA
inat_resnet.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(inat_resnet.classifier.parameters())

# %%
def test_cub(model):
  model.eval()
  
  running_loss = 0.0
  running_corrects = 0
  
  predictions = []
  targets = []
  confidence = []
  
  class_corrects = [0] * 200
  class_counts = [0] * 200
  
  full_paths = []
  with torch.inference_mode():
    for _, (data, target, path) in tqdm(enumerate(val_loader)):
      data   = data.to(device)
      target = target.to(device)
      outputs = model(data)
      loss = criterion(outputs, target)
      _, preds = torch.max(outputs, 1)
      probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
      running_loss += loss.item() * target.size(0)
      running_corrects += torch.sum(preds == target.data)

      # class accuracy
      for i in range(len(target)):
        class_corrects[target[i]] += int(preds[i] == target[i])
        class_counts[target[i]] += 1
      ####
      predictions.extend(preds.data.cpu().numpy())
      targets.extend(target.data.cpu().numpy())

      bird_labels = target.detach().to('cpu').tolist()
      bird_preds = preds.detach().to('cpu').tolist()
      for i, (label, pred, _path) in enumerate(zip(bird_labels, bird_preds, path)):
        if pred == label:
          full_paths.append(_path.split('/')[-2] + '/' + _path.split('/')[-1])

      confidence.extend((probs.data.cpu().numpy()*100).astype(np.int32))

    class_accuracies = [correct / count if count != 0 else 0 for correct, count in zip(class_corrects, class_counts)]

  epoch_loss = running_loss / len(validation_folder)
  epoch_acc = running_corrects.double() / len(validation_folder)

  print('-' * 10)
  print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100*epoch_acc))
  
  return predictions, targets, confidence, full_paths, class_accuracies

# %%
cub_test_preds, cub_test_targets, cub_test_confs, full_paths, class_acc = test_cub(inat_resnet)

# %% save class accuracy
import csv
class_to_idx = validation_folder.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
sup_type = 'mohammad'
csv_file_path = f"/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/{sup_type}_class_accuracies.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Class", "Accuracy"])
    for class_idx, accuracy in enumerate(class_acc):
        csv_writer.writerow([idx_to_class[class_idx], accuracy])
# %%
print(len(full_paths))
def save_paths_to_txt(file_path, paths_list):
  with open(file_path, 'w') as file:
    for path in paths_list:
      file.write(f"{path}\n")

save_paths_to_txt('./mohammad_paths.txt', full_paths)
exit()

# %%
# ensemble with habitat model
import timm
import clip

habitat_model_type = 'resnet' # or resnet

if habitat_model_type == 'resnet':

  habitat_model = timm.create_model(
              'resnet101',
              pretrained=True,
              num_classes=200,
              in_chans=3,
          )
  habitat_model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/cub-4-0.079-resnet101-inpaint_True.pth'
elif habitat_model_type == 'clip':
  clip_model, transform = clip.load('ViT-L/14', device=device)
    
  visual_encoder = clip_model.visual
  visual_encoder.fc = nn.Identity()

  for param in visual_encoder.parameters():
      param.requires_grad = False
    
  habitat_model = nn.Sequential(
                visual_encoder,
                nn.ReLU(),
                nn.Linear(visual_encoder.output_dim, 200)
                ).to(torch.float32)
  habitat_model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/16_cub_clip_0.104.pth'

elif habitat_model_type == 'resnet_fix':
  habitat_model = timm.create_model('resnet101', pretrained=True)
  n_inputs = habitat_model.fc.in_features
  habitat_model.fc = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 200))
  habitat_model_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/cub-7-resnet101-inpaint.pth'
  
# habitat_model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
if habitat_model_type == 'resnet_fix':
  habitat_model = torch.jit.load(habitat_model_path)
else:
  habitat_model.load_state_dict(torch.load(habitat_model_path))

habitat_model.to(device)

# %%
# # ensemble
# def test_cub_ensemble(id_model, habitat_model, alpha=1):
#   id_model.eval()
#   habitat_model.eval()
  
#   id_model_1 = torch.nn.Sequential(*list(id_model.children())[:-1])
#   id_model_2 = id_model.classifier

#   habitat_model_1 = torch.nn.Sequential(*list(habitat_model.children())[:-1])
#   # habitat_model_1 = torch.nn.Sequential(*list(habitat_model.visual.children())[:-1])
#   # print(habitat_model_1)
#   # habitat_model_2 = habitat_model.classifier
  
#   running_loss = 0.0
#   running_corrects = 0
  
#   predictions = []
#   confidence = []
  
#   with torch.inference_mode():
#     for _, (data, target) in enumerate(val_loader):
#       data   = data.to(device)
#       target = target.to(device)
      
#       id_pooling_features = id_model_1(data)
#       habitat_pooling_features = habitat_model_1(data)
#       ensemble_pooling_features = alpha*id_pooling_features.squeeze() + (1-alpha)*habitat_pooling_features.squeeze()
#       outputs = id_model_2(ensemble_pooling_features)
#       # outputs = outputs.unsqueeze(0)

#       loss = criterion(outputs, target)
#       _, preds = torch.max(outputs, 1)
#       probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
#       running_loss += loss.item() * target.size(0)
#       running_corrects += torch.sum(preds == target.data)
      
#       predictions.extend(preds.data.cpu().numpy())
#       confidence.extend((probs.data.cpu().numpy()*100).astype(np.int32))

#   epoch_loss = running_loss / len(validation_folder)
#   epoch_acc = running_corrects.double() / len(validation_folder)

#   print('-' * 10)
#   print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100*epoch_acc))
  
#   return predictions, confidence

# %%
# cub_test_preds, cub_test_confs = test_cub_ensemble(inat_resnet, habitat_model, alpha=0.5)

# %%
# ensemble
def test_cub_ensemble_2(id_model, habitat_model, alpha=1):
  id_model.eval()
  habitat_model.eval()
  
  running_loss = 0.0
  running_corrects = 0
  
  predictions = []
  targets = []
  paths = []
  confidence = []
  
  with torch.inference_mode():
    for _, (data, data2, target, path, path2) in enumerate(val_loader):
      data   = data.to(device)
      target = target.to(device)
      
      id_feat = id_model(data)
      habitat_feat = habitat_model(data)
      outputs = id_feat*alpha + habitat_feat*(1-alpha)
      # outputs = outputs.unsqueeze(0)

      loss = criterion(outputs, target)
      _, preds = torch.max(outputs, 1)
      probs, _ = torch.max(F.softmax(outputs, dim=1), 1)
      running_loss += loss.item() * target.size(0)
      running_corrects += torch.sum(preds == target.data)
      
      predictions.extend(preds.data.cpu().numpy())
      targets.extend(target.data.cpu().numpy())
      paths.extend(path)
      confidence.extend((probs.data.cpu().numpy()*100).astype(np.int32))

  epoch_loss = running_loss / len(validation_folder)
  epoch_acc = running_corrects.double() / len(validation_folder)

  print('-' * 10)
  print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100*epoch_acc))
  
  return predictions, targets, confidence, paths

# %%
def test_cub_ensemble_3(id_model, habitat_model, alpha=1):
  id_model.eval()
  habitat_model.eval()
  
  running_loss = 0.0
  running_corrects = 0
  
  predictions = []
  targets = []
  paths = []
  confidence = []
  
  with torch.inference_mode():
    for _, (data1, data2, target, path, path2) in enumerate(val_loader):
      data1   = data1.to(device)
      data2   = data2.to(device)
      target = target.to(device)
      
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
      paths.extend(path)
      confidence.extend((probs.data.cpu().numpy()*100).astype(np.int32))

  epoch_loss = running_loss / len(validation_folder)
  epoch_acc = running_corrects.double() / len(validation_folder)

  print('-' * 10)
  print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, 100*epoch_acc))
  
  return predictions, targets, confidence, paths
# %%
# cub_test_ensemble_preds, cub_test_ensemble_targets, cub_test_ensemble_confs, img_paths = test_cub_ensemble_2(inat_resnet, habitat_model, alpha=0.7)


# %%
# import os, shutil
# wrong2correct_paths = []
# correct2wrong_paths = []
# wrong2wrong_paths = []
# for orig_pred, ensemble_pred, target, path in zip(cub_test_preds, cub_test_ensemble_preds, cub_test_ensemble_targets, img_paths):
#     if orig_pred != target and ensemble_pred == target:
#         wrong2correct_paths.append(path)
#     if orig_pred == target and ensemble_pred != target:
#         correct2wrong_paths.append(path)
#     if orig_pred != target and ensemble_pred != target:
#         wrong2wrong_paths.append(path)
# print(len(wrong2correct_paths))
# print(len(correct2wrong_paths))
# print(len(wrong2wrong_paths))
# if not os.path.exists('./wrong2correct/'):
#     os.makedirs('./wrong2correct/')
# if not os.path.exists('./correct2wrong/'):
#     os.makedirs('./correct2wrong/')
# if not os.path.exists('./wrong2wrong/'):
#     os.makedirs('./wrong2wrong/')

# for img_path in wrong2correct_paths:
#     shutil.copy(img_path, './wrong2correct/')
# for img_path in correct2wrong_paths:
#     shutil.copy(img_path, './correct2wrong/')
# for img_path in wrong2wrong_paths:
#     shutil.copy(img_path, './wrong2wrong/')

# %%
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(alpha)
    cub_test_preds, _, cub_test_confs, _ = test_cub_ensemble_3(inat_resnet, habitat_model, alpha=alpha)

# %%



