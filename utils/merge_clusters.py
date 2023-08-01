# %%
import os
import cv2
import numpy as np
import math
import pandas as pd
import json
# %%
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_thickness = 1

def merge_images(imgs, descs):
    """
    imgs: numpy images
    descs: descs of images
    """
    if len(imgs) == 0:
        print(descs)
    if len(imgs) == 1:
        return imgs[0]
    # for img, desc in zip(imgs, descs):
    #     images = []
    
    query_image = np.concatenate(imgs, axis=1)
    

    return query_image

# %%

cluster_classes_file = '/home/tin/projects/reasoning/plain_clip/class_clusters.json'
f = open(cluster_classes_file)
cluster_classes = json.load(f)

def has_duplicate_in_other_lists(list_of_lists, target_list_index):
    target_list = list_of_lists[target_list_index]

    for index, current_list in enumerate(list_of_lists):
        if index == target_list_index:
            continue  # Skip the target list itself

        for element in target_list:
            if element in current_list:
                return True

    return False

# Check if elements of list at index 0 are duplicated in other lists
for i in range(101):
    result = has_duplicate_in_other_lists(list(cluster_classes.values()), i)
    if result:
        print(i)
    



# %%
description_path = "../plain_clip/descriptors/cub/additional_chatgpt_descriptors_cub.json"
f = open(description_path, 'r')
documents = json.load(f)
documents = {k: v[-1][9:] for k,v in documents.items()}

# %%

save_folder = "merged_clusters/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

images_folder = 'merged_cub/'
images_names = os.listdir(images_folder)

for idx, classes in cluster_classes.items():
    descs = [documents[cls] for cls in classes]
    imgs = []
    for cls in classes:
        for orig_name in images_names:
            name = orig_name[:-4].split('.')[1]
            if len(name.split('_')) > 2:
                name_parts = name.split('_')
                if len(name.split('_')) == 3:
                    name = name_parts[0] + '-' + name_parts[1] + ' ' + name_parts[2]
                else:
                    name = name_parts[0] + '-' + name_parts[1] + '-' + name_parts[2] + ' ' + name_parts[3]
            else:
                name = name.replace('_', ' ')
            if cls == name:
                img = cv2.imread(f"{images_folder}/{orig_name}")
                img = cv2.resize(img, (1120, 2976))
                imgs.append(img)

    
    
    image = merge_images(imgs, descs)
    cv2.imwrite(f"{save_folder}/{idx}.jpg", image)
    
# %%
