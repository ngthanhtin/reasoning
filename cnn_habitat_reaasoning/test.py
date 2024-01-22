# %%
path1 = 'mohammad_paths.txt'
path2 = 'tin_paths.txt'

# %%
def read_image_paths(file_path):
    with open(file_path, 'r') as file:
        image_paths = [line.strip() for line in file]
    return image_paths

image_paths_list1 = read_image_paths(path1)
image_paths_list2 = read_image_paths(path2)

print("Image paths in file1.txt:")
print(image_paths_list1[:5])

print("\nImage paths in file2.txt:")
print(image_paths_list2[:5])
# %%
non_overlapping_image_paths = set(image_paths_list1).symmetric_difference(set(image_paths_list2))
non_overlapping_image_paths_list = list(non_overlapping_image_paths)
len(non_overlapping_image_paths_list)
# %%
non_overlapping_image_paths_list[:5]
# %%
orig_root = '/home/tin/datasets/cub/CUB/images/'
inpaint_root = '/home/tin/datasets/cub/CUB_inpaint_all/'
save_folder = 'concat_images/'

import os
import cv2
for path in non_overlapping_image_paths_list:
    img_name = path.split('/')[-1]
    orig_path = os.path.join(orig_root, path)
    inpaint_path = os.path.join(inpaint_root, path)

    img1 = cv2.imread(orig_path)
    img2 = cv2.imread(inpaint_path)

    concatenated_horizontal = cv2.hconcat([img1, img2])

    cv2.imwrite(f'{save_folder}/{img_name}', concatenated_horizontal)

# %%
import torch

# Create a tensor with shape (batch_size, dim)
batch_size = 3
dim = 4
tensor = torch.tensor([[1, 2, 1, 1],
                       [3, 3, 3, 3],
                       [0, 0, 0, 0]])

tensor = torch.tensor([[1, 1, 1, 1],
                       [1, 1, 3, 3],
                       [0, 0, 0, 0]])

# Value to check against
value_to_check = 1

# Check if all values in each row of the tensor are equal to the specified value
are_all_equal_per_row = torch.all(tensor == value_to_check, dim=1)

print(are_all_equal_per_row.tolist())
# %%
# path = './weight_figures_4/'
path = '/home/tin/datasets/nabirds/flybird_nabirds_test/'
import os
folder = os.listdir(path)
num = 0
for fol in folder:
    folder_path = path + fol
    num+=len(os.listdir(folder_path))

num

# For NABirds: train: 23929, test: 24633
# For mask: 48562
# %%
import os, cv2
import numpy as np
inpaint_folder = '/home/tin/datasets/cub/CUB_no_bg_train/'
save_folder = '/home/tin/datasets/cub/CUB_blank_train/'

label_folders = os.listdir(inpaint_folder)
for folder in label_folders:
    if not os.path.exists(f"{save_folder}/{folder}"):
        os.makedirs(f"{save_folder}/{folder}")
    
    image_files = os.listdir(f"{inpaint_folder}/{folder}")
    for file in image_files:
        image_path = f"{inpaint_folder}/{folder}/{file}"
        img = cv2.imread(image_path)
        blank_img = np.zeros_like(img)

        cv2.imwrite(f"{save_folder}/{folder}/{file}", blank_img)

# %% show segmentation CUB images
import os, cv2
import numpy as np
segment_folder = '/home/tin/datasets/cub/CUB/segmentations/'
label_folders = os.listdir(segment_folder)

for folder in label_folders:
    image_files = os.listdir(f"{segment_folder}/{folder}")
    for i1, file1 in enumerate(image_files):
        onlybird_image_path = f"{segment_folder}/{folder}/{file1}"
        onlybird_img = cv2.imread(onlybird_image_path)
        onlybird_img[onlybird_img != 255] = 0
        cv2.imwrite(f"./test_aug/{file1}", onlybird_img)

        if i1 == 10:
            break
    break
# %%
def get_mask_object_bbox(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    y, y_end = np.where(rows)[0][[0, -1]]
    x, x_end = np.where(cols)[0][[0, -1]]
    width = x_end - x + 1
    height = y_end - y + 1
    return x, y, width, height

def mask_rectangle(image, x, y, width, height):
    image[y:y+height, x:x+width] = 1

# %% Make augmentation dataset
import os, cv2
import numpy as np
orig_folder = '/home/tin/datasets/cub/CUB/train/'
inpaint_folder = '/home/tin/datasets/cub/CUB_inpaint_all_train/'
onlybird_folder = '/home/tin/datasets/cub/CUB_no_bg_train/'
save_folder = '/home/tin/datasets/cub/CUB_aug_train_rect/'
segment_folder = '/home/tin/datasets/cub/CUB/segmentations/'

label_folders = os.listdir(inpaint_folder)
for folder in label_folders:
    if not os.path.exists(f"{save_folder}/{folder}"):
        os.makedirs(f"{save_folder}/{folder}")
    
    image_files = os.listdir(f"{inpaint_folder}/{folder}")
    for i1, file1 in enumerate(image_files):
        onlybird_image_path = f"{orig_folder}/{folder}/{file1}"
        onlybird_img = cv2.imread(onlybird_image_path)

        mask_path = f"{segment_folder}/{folder}/{file1[:-4]}.png"
        mask_img = cv2.imread(mask_path)

        for i2, file2 in enumerate(image_files):
            inpaint_image_path = f"{inpaint_folder}/{folder}/{file2}"
            inpaint_img = cv2.imread(inpaint_image_path)
            # resize
            resize_onlybird_img = cv2.resize(onlybird_img, (inpaint_img.shape[1], inpaint_img.shape[0]))
            mask_img = cv2.resize(mask_img, (inpaint_img.shape[1], inpaint_img.shape[0]))
            # mask_img[mask_img != 255] = 0
            x,y,w,h = get_mask_object_bbox(mask_img)

            # inpaint_img[mask_img != 0] = 0
            # final_image = resize_onlybird_img + inpaint_img
            inpaint_img[x:x+w,y:y+h] = 0
            inpaint_img[x:x+w, y:y+h] = resize_onlybird_img[x:x+w, y:y+h]
            final_image = inpaint_img
            # cv2.imwrite(f"./test_aug/abc_{i1}_{i2}.png", inpaint_img)
            
    #     break
    # break
            
            cv2.imwrite(f"{save_folder}/{folder}/{file1[:-4]}_{file2}", final_image)
# %%
import random
import numpy as np
np.random.seed(45)
def get_random_subset(input_list, subset_size, restrict_files=[]):
    if subset_size > len(input_list):
        raise ValueError("Subset size cannot be greater than the length of the input list.")
    
    i = 0
    random_subset = []
    while i < 30:
        random_file = random.sample(input_list, 1)[0]
        random_file_name = random_file[:-4]
        is_in = False
        for restrict_file in restrict_files:
            if random_file_name in restrict_file:
                is_in = True
            if is_in:
                break
        if not is_in:
            random_subset.append(random_file)
            i += 1

    return random_subset

#%%
import os, cv2
import shutil
import numpy as np
import random
from tqdm import tqdm

# orig_folder = '/home/tin/datasets/cub/CUB_irrelevant_augmix_train/'
# save_folder = '/home/tin/datasets/cub/temp_gen_data/CUB_aug_irrelevant_with_orig_birds_train_60/'

orig_folder = '/home/tin/datasets/nabirds/gen_data/augmix_images/'
save_folder = '/home/tin/datasets/nabirds/gen_data/augmix_images_small_diff_30_added_1/'
flybird_folder = '/home/tin/datasets/nabirds/flybird_nabirds_train/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

label_folders = os.listdir(orig_folder)
subset_size = 30
for folder in tqdm(label_folders):
    if not os.path.exists(f"{save_folder}/{folder}"):
        os.makedirs(f"{save_folder}/{folder}")
    image_files = os.listdir(f"{orig_folder}/{folder}")
    flybird_files = os.listdir(f"{flybird_folder}/{folder}")
    
    if len(image_files) < subset_size:
        random_subset = get_random_subset(image_files, len(image_files))
    else:
        random_subset = get_random_subset(image_files, subset_size, restrict_files = flybird_files)
    for file in random_subset:
        image_path = f"{orig_folder}/{folder}/{file}"
        shutil.copy(image_path, f"{save_folder}/{folder}/{file}")
# %%
import os, cv2
import shutil
import numpy as np
from tqdm import tqdm
# orig_folder = '/home/tin/datasets/cub/temp_gen_data/CUB_irrelevant_with_orig_birds_train/'
# save_folder = '/home/tin/datasets/cub/temp_gen_data/CUB_aug_irrelevant_with_orig_birds_train_60/'
orig_folder = '/home/tin/datasets/nabirds/train/'
save_folder = '/home/tin/datasets/nabirds/gen_data/augmix_images_small_diff_30_added_1/'
label_folders = os.listdir(orig_folder)
for folder in tqdm(label_folders):
    
    image_files = os.listdir(f"{orig_folder}/{folder}")
    for file in image_files:
        image_path = f"{orig_folder}/{folder}/{file}"
        shutil.copy(image_path, f"{save_folder}/{folder}/{file}")
# %% make CUB_random into CUB_random_train/ CUB_random_test/

path = '/home/tin/datasets/cub/CUB_random/'
train_path = '/home/tin/datasets/cub/CUB_random_train/'
test_path = '/home/tin/datasets/cub/CUB_random_test/'

import os, shutil
label_folders = os.listdir(path)

# read train_test_split
# obtain sample ids filtered by split
path_to_splits = os.path.join('/home/tin/datasets/cub/CUB/', 'train_test_split.txt')
indices_to_use = list()
with open(path_to_splits, 'r') as in_file:
    for line in in_file:
        idx, use_train = line.strip('\n').split(' ', 2)
        if bool(int(use_train)) == 0:
            indices_to_use.append(int(idx))

# obtain filenames of images
path_to_index = os.path.join('/home/tin/datasets/cub/CUB/', 'images.txt')
filenames_to_use = set()
with open(path_to_index, 'r') as in_file:
    for line in in_file:
        idx, fn = line.strip('\n').split(' ', 2)
        if int(idx) in indices_to_use:
            filenames_to_use.add(fn)

# %%
c = 0
for label_f in label_folders:
    folder_path = f'/home/tin/datasets/cub/CUB_random/{label_f}'
    image_files = os.listdir(folder_path)
    for file in image_files:
        if f'{label_f}/{file}' in filenames_to_use:
            c += 1
            image_path = f"/home/tin/datasets/cub/CUB_random/{label_f}/{file}"
            if not os.path.exists(f"/home/tin/datasets/cub/CUB_random_test/{label_f}"):
                os.makedirs(f"/home/tin/datasets/cub/CUB_random_test/{label_f}")
            save_path = f"/home/tin/datasets/cub/CUB_random_test/{label_f}/{file}"
            shutil.copy(image_path, save_path)



# %% NABirds
import os, cv2
import numpy as np
orig_folder = '/home/tin/datasets/nabirds/images/'
segment_folder = '/home/tin/datasets/nabirds/gen_data/mask_images/'
save_folder = '/home/tin/datasets/nabirds/gen_data/onlybird_images/'
label_folders = os.listdir(segment_folder)

for folder in label_folders:
    image_files = os.listdir(f"{segment_folder}/{folder}")
    for i1, file1 in enumerate(image_files):
        onlybird_image_path = f"{segment_folder}/{folder}/{file1}"
        onlybird_img = cv2.imread(onlybird_image_path)

        orig_image_path = f"{orig_folder}/{folder}/{file1}"
        orig_img = cv2.imread(orig_image_path)
        # orig_img[onlybird_img != 255] = 0
        cv2.imwrite(f"./test_aug/{file1}", onlybird_img)

        if i1 == 10:
            break
    break

# %%
orig_folder = '/home/tin/datasets/cub/CUB/train/'
inpaint_folder = '/home/tin/datasets/cub/CUB_inpaint_all_train/'
onlybird_folder = '/home/tin/datasets/cub/CUB_no_bg_train/'
save_folder = '/home/tin/datasets/cub/CUB_aug_train_rect/'
segment_folder = '/home/tin/datasets/cub/CUB/segmentations/'

label_folders = os.listdir(inpaint_folder)
for folder in label_folders:
    if not os.path.exists(f"{save_folder}/{folder}"):
        os.makedirs(f"{save_folder}/{folder}")
    
    image_files = os.listdir(f"{inpaint_folder}/{folder}")
    for i1, file1 in enumerate(image_files):
        onlybird_image_path = f"{orig_folder}/{folder}/{file1}"
        onlybird_img = cv2.imread(onlybird_image_path)

        mask_path = f"{segment_folder}/{folder}/{file1[:-4]}.png"
        mask_img = cv2.imread(mask_path)

        for i2, file2 in enumerate(image_files):
            inpaint_image_path = f"{inpaint_folder}/{folder}/{file2}"
            inpaint_img = cv2.imread(inpaint_image_path)
            # resize
            resize_onlybird_img = cv2.resize(onlybird_img, (inpaint_img.shape[1], inpaint_img.shape[0]))
            mask_img = cv2.resize(mask_img, (inpaint_img.shape[1], inpaint_img.shape[0]))
            # mask_img[mask_img != 255] = 0
            x,y,w,h = get_mask_object_bbox(mask_img)

            # inpaint_img[mask_img != 0] = 0
            # final_image = resize_onlybird_img + inpaint_img
            inpaint_img[x:x+w,y:y+h] = 0
            inpaint_img[x:x+w, y:y+h] = resize_onlybird_img[x:x+w, y:y+h]
            final_image = inpaint_img
            # cv2.imwrite(f"./test_aug/abc_{i1}_{i2}.png", inpaint_img)
            
    #     break
    # break
            
            cv2.imwrite(f"{save_folder}/{folder}/{file1[:-4]}_{file2}", final_image)

# %% Apply Coarse Dropout to Image
import albumentations as A
import matplotlib.pyplot as plt
import os, cv2
from PIL import Image
import numpy as np

# get the mask object bbox
def get_mask_object_bbox(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    y, y_end = np.where(rows)[0][[0, -1]]
    x, x_end = np.where(cols)[0][[0, -1]]
    width = x_end - x + 1
    height = y_end - y + 1
    return x, y, width, height

def dropout_with_mask(image, mask, num_boxes=8, SZ=0.1):
    # numpy image
    # input - image of size [dim,dim,3], mask of size [dim,dim]
    # output - image with num_boxes squares 

    h,w = image.shape[:2]
    mask_x, mask_y, mask_w, mask_h = get_mask_object_bbox(mask) #x,y,w,h
    any_values_along_rows = np.any(mask, axis=1)
    count_mask = np.count_nonzero(any_values_along_rows)
    
    k=0
    while True:
        # CHOOSE RANDOM LOCATION
        x = int(np.random.uniform(mask_x, mask_x + mask_w/4))
        y = int(np.random.uniform(mask_y, mask_y + mask_h/4))

        # COMPUTE SQUARE
        WIDTH = int(SZ * mask_w)
        HEIGHT = int(SZ * mask_h)
        ya = y#max(0, y - HEIGHT // 2)
        yb = y+HEIGHT#min(h, y + HEIGHT // 2)
        xa = x#max(0, x - WIDTH // 2)
        xb = x+WIDTH#min(w, x + WIDTH // 2)

        # Apply dropout only where the mask is active
        if np.any(mask[ya:yb, xa:xb]):
            any_values_along_rows = np.any(mask[ya:yb, xa:xb], axis=1)
            count_overlapping = np.count_nonzero(any_values_along_rows)
            
            if count_overlapping/count_mask>0.15:   
                # DROPOUT IMAGE
                image[ya:yb, xa:xb, :] = 0.
                k+=1
        if k == num_boxes:
            break

    return image

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    # plt.show()
    plt.savefig("squares.png")

cub_path = '/home/tin/datasets/cub/CUB/test/' #'/home/tin/datasets/nabirds/test/'#'/home/tin/datasets/cub/CUB/test/'
save_path = '/home/tin/datasets/cub/CUB_big_bb_on_birds_test/' #'/home/tin/datasets/nabirds/gen_data/big_bb_on_birds_test/'#'/home/tin/datasets/cub/CUB_bb_on_birds_test/'
segment_path = '/home/tin/datasets/cub/CUB/segmentations/' #'/home/tin/datasets/nabirds/gen_data/mask_images/' #'/home/tin/datasets/cub/CUB/segmentations/'

label_folders = os.listdir(cub_path)
from tqdm import tqdm
num_failed = 0
for folder in tqdm(label_folders):
    if not os.path.exists(f"{save_path}/{folder}"):
        os.makedirs(f"{save_path}/{folder}")

    image_files = os.listdir(f"{cub_path}/{folder}")
    segment_files = os.listdir(f"{segment_path}/{folder}")

    for _, file in enumerate(image_files):
        if os.path.exists(f"{save_path}/{folder}/{file}"):
            continue
        image_path = f"{cub_path}/{folder}/{file}"
        # mask_path = f"{segment_path}/{folder}/{file}"
        mask_path = f"{segment_path}/{folder}/{file[:-4]}.png"

        img = cv2.imread(image_path)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask.astype(float)/255
        # mask = mask.astype(np.uint8)
        try:
            dropout_img = dropout_with_mask(img, mask, num_boxes=1, SZ=0.5)
        except:
            num_failed += 1
            dropout_img = img
        
        cv2.imwrite(f"{save_path}/{folder}/{file}", dropout_img)
num_failed

# %%
import numpy as np

arr = np.array([[False, False, False],
                [False, True, False],
                [False, False, True]])

# Apply np.any along rows (axis=1)
any_values_along_rows = np.any(arr, axis=0)

# Count the number of overlapping True values along rows
count_overlapping = np.count_nonzero(any_values_along_rows)

print("Number of overlapping True values along rows:", count_overlapping)
# %%
import numpy as np
from scipy import stats

# Sample confidences from two sets (replace these with your actual data)
confidences_set1 = np.array([0.85, 0.78, 0.92, 0.65, 0.75])
confidences_set2 = np.array([0.80, 0.72, 0.88, 0.68, 0.81])

# Compute the mean and standard deviation of each set
mean_set1 = np.mean(confidences_set1)
std_dev_set1 = np.std(confidences_set1, ddof=1)  # Use ddof=1 for sample standard deviation

mean_set2 = np.mean(confidences_set2)
std_dev_set2 = np.std(confidences_set2, ddof=1)

# Calculate the confidence intervals for each set
confidence_level = 0.95
conf_interval_set1 = stats.norm.interval(confidence_level, loc=mean_set1, scale=std_dev_set1)
conf_interval_set2 = stats.norm.interval(confidence_level, loc=mean_set2, scale=std_dev_set2)

print(conf_interval_set1)
print(conf_interval_set2)
# Calculate the overlap between the confidence intervals
overlap = max(0, min(conf_interval_set1[1], conf_interval_set2[1]) - max(conf_interval_set1[0], conf_interval_set2[0]))

print(f"Confidence interval overlap: {overlap:.2f}")



# %% check overlap between CUB and INaturalist


import os, json
import Levenshtein

def find_overlapping_with_distance(set1, set2, set1_folders, set2_folders, threshold=2.):
    overlapping_names_triple = []

    for name1 in set1:
        for sci_name2, comm_name2 in set2.items():
            if name1.lower() == comm_name2.lower():
                overlapping_names_triple.append([name1, comm_name2, sci_name2, str(set1_folders[name1]), set2_folders[sci_name2]])
                break
            # distance = Levenshtein.distance(name1, name2)
            # if distance <= threshold:
                # overlapping_names_pair.append([name1, name2])
                # break
                # overlapping_names.add(name2)

    return overlapping_names_triple

inat_sci2comm = 'inaturalist_sci2comm.json'
f = open(inat_sci2comm)
inat_sci2comm_data = json.load(f)

with open(os.path.join('/home/tin/datasets/inaturalist2021_onlybird/bird_classes.json'), 'r') as f:
    class_meta = json.load(f)
    # easy mapping for class id, class name and corresponding folder name
    
idx2class = {int(id): name for id, name in class_meta['name'].items()} # if use subset
class2idx = {v: k for k, v in idx2class.items()}

cls_id2folder_name = {int(id): name for id, name in class_meta['image_dir_name'].items()} # if use subset
folder_name2cls_id = {v: k for k, v in cls_id2folder_name.items()}

inat_class2folder_name = {cls: cls_id2folder_name[id] for cls, id in class2idx.items()}
inat_class2folder_name, len(inat_class2folder_name)
# %% read cub class names
cub_path = '/home/tin/datasets/cub/CUB/train/'
cub_label_folders = os.listdir(cub_path)
cub_labelfolder2class = {}
for label_folder in cub_label_folders:
    name = label_folder.split('.')[1]
    
    if len(name.split('_')) > 2:
        name_parts = name.split('_')
        if len(name.split('_')) == 3:
            name = name_parts[0] + '-' + name_parts[1] + ' ' + name_parts[2]
        else:
            name = name_parts[0] + '-' + name_parts[1] + '-' + name_parts[2] + ' ' + name_parts[3]
    else:
        name = name.replace('_', ' ')
    cub_labelfolder2class[label_folder] = name

cub_class2labelfolder = {v:k for k,v in cub_labelfolder2class.items()}
list_cub_labels = [v for k,v in cub_labelfolder2class.items()]

# %%
threshold=2
cub_overlapping_names_triple = find_overlapping_with_distance(list_cub_labels, inat_sci2comm_data, cub_class2labelfolder, inat_class2folder_name, threshold)
cub_overlapping_names_triple, len(cub_overlapping_names_triple)
# %%
import csv
csv_file_path = "cub_inat_triple_exact_match.csv"

# Write the list of pairs to the CSV file
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["cub", "comm_inat", "sci_inat"])  # Write header
    csv_writer.writerows(cub_overlapping_names_triple)          

# %% read nabirds classname
import pandas as pd
def read_hierarchy(bird_dir='/home/tin/datasets/nabirds/'):
    """Loads table of class hierarchies. Returns hierarchy table
    parent-child class map, top class levels, and bottom class levels.
    """
    hierarchy = pd.read_table(f'{bird_dir}/hierarchy.txt', sep=' ',
                              header=None)
    hierarchy.columns = ['child', 'parent']

    child_graph = {0: []}
    name_level = {0: 0}
    for _, row in hierarchy.iterrows():
        child_graph[row['parent']].append(row['child'])
        child_graph[row['child']] = []
        name_level[row['child']] = name_level[row['parent']] + 1
    
    terminal_levels = set()
    for key, value in name_level.items():
        if not child_graph[key]:
            terminal_levels.add(key)

    parent_map = {row['child']: row['parent'] for _, row in hierarchy.iterrows()}
    return hierarchy, parent_map, set(child_graph[0]), terminal_levels

hierarchy, parent_map, _, terminal_levels = read_hierarchy()
discrete_labels = set(hierarchy.parent.values.tolist())

def read_class_labels(top_levels, parent_map, bird_dir='/home/tin/datasets/nabirds/'):
    """Loads table of image IDs and labels. Add top level ID to table."""
    def get_class(l):
        return l if l in top_levels else get_class(parent_map[l])

    class_labels = pd.read_table(f'{bird_dir}/image_class_labels.txt', sep=' ',
                                 header=None)
    class_labels.columns = ['image', 'id']
    class_labels['class_id'] = class_labels['id'].apply(get_class)

    return class_labels

class_labels = read_class_labels(terminal_levels, parent_map)

def read_classes(terminal_levels, bird_dir='/home/tin/datasets/nabirds/'):
    """Loads DataFrame with class labels. Returns full class table
    and table containing lowest level classes.
    """
    def make_annotation(s):
        try:
            return s.split('(')[1].split(')')[0]
        except Exception as e:
            return None

    classes = pd.read_table('/home/tin/projects/reasoning/scraping/nabird_data/nabird_classes.txt', header=None) # this file does not have double spaces
    classes['id'] = classes[0].apply(lambda s: int(s.split(' ')[0]))
    classes['label_name'] = classes[0].apply(lambda s: ' '.join(s.split(' ')[1:]))
    classes.drop(0, inplace=True, axis=1)
    classes['annotation'] = classes['label_name'].apply(make_annotation)
    classes['name'] = classes['label_name'].apply(lambda s: s.split('(')[0].strip())

    terminal_classes = classes[classes['id'].isin(terminal_levels)]#.reset_index(drop=True)
    return classes, terminal_classes

nabirds_classes, nabirds_terminal_classes = read_classes(terminal_levels)
labelname2labelidx = nabirds_terminal_classes.set_index('label_name')['id'].to_dict()

print(len(labelname2labelidx), labelname2labelidx)
list_nabirds_labels = [k for k, v in labelname2labelidx.items()]
# %%
nabirds_overlapping_names_triple = find_overlapping_with_distance(list_nabirds_labels, inat_sci2comm_data, labelname2labelidx, inat_class2folder_name, threshold)
nabirds_overlapping_names_triple, len(nabirds_overlapping_names_triple)

# %%
import csv
csv_file_path = "nabirds_inat_triple_exact_match.csv"

# Write the list of pairs to the CSV file
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["nabirds", "comm_inat", "sci_inat"])  # Write header
    csv_writer.writerows(nabirds_overlapping_names_triple)    
# %%
import shutil, os
from tqdm import tqdm
cub_overlapping_names_triple
nabirds_overlapping_names_triple

inat_path = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/"
save_cub_inat_overlapping_path = '/home/tin/datasets/overlapping_cub_inat/'
save_nabirds_inat_overlapping_path = '/home/tin/datasets/overlapping_nabirds_inat/'

cub_path = '/home/tin/datasets/cub/CUB/test/'
cub_label_folders = os.listdir(cub_path)
for folder in tqdm(cub_label_folders):
    is_overlap = False
    os.makedirs(f"{save_cub_inat_overlapping_path}/{folder}", exist_ok=True)
    for overlapping in cub_overlapping_names_triple:
        if folder == overlapping[3]:
            is_overlap = True
            # copy inat images to created folder
            inat_folder_path = f"{inat_path}/{overlapping[4]}"
            image_files = os.listdir(inat_folder_path)
            image_paths = [inat_folder_path + f'/{file}' for file in image_files]
            for p in image_paths:
                shutil.copy(p, f"{save_cub_inat_overlapping_path}/{folder}")
            break
    if not is_overlap:
        #copy cub images to created folder
        cub_folder_path = f'{cub_path}/{folder}'
        image_files = os.listdir(cub_folder_path)
        image_paths = [cub_folder_path + f'/{file}' for file in image_files]
        for p in image_paths:
            shutil.copy(p, f"{save_cub_inat_overlapping_path}/{folder}")

# %%
# import shutil, os
# nabirds_overlapping_names_triple

# inat_path = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/"
# save_nabirds_inat_overlapping_path = '/home/tin/datasets/overlapping_nabirds_inat/'

# nabirds_path = '/home/tin/datasets/nabirds/test/'
# nabirds_label_folders = os.listdir(nabirds_path)
# from tqdm import tqdm
# for folder in tqdm(nabirds_label_folders):
#     is_overlap = False
#     os.makedirs(f"{save_nabirds_inat_overlapping_path}/{folder}", exist_ok=True)
#     for overlapping in nabirds_overlapping_names_triple:
#         if int(folder) == int(overlapping[3]):
#             is_overlap = True
#             # copy inat images to created folder
#             inat_folder_path = f"{inat_path}/{overlapping[4]}"
#             image_files = os.listdir(inat_folder_path)
#             image_paths = [inat_folder_path + f'/{file}' for file in image_files]
#             for p in image_paths:
#                 if os.path.exists(f"{save_nabirds_inat_overlapping_path}/{folder}/{p}"):
#                     continue
#                 shutil.copy(p, f"{save_nabirds_inat_overlapping_path}/{folder}")
#             break
#     if not is_overlap:
#         #copy nabirds images to created folder
#         nabirds_folder_path = f'{nabirds_path}/{folder}'
#         image_files = os.listdir(nabirds_folder_path)
#         image_paths = [nabirds_folder_path + f'/{file}' for file in image_files]
#         for p in image_paths:
#             if os.path.exists(f"{save_nabirds_inat_overlapping_path}/{folder}/{p}"):
#                 continue
#             shutil.copy(p, f"{save_nabirds_inat_overlapping_path}/{folder}")
# %%

# %% test accs of 144 cub-inat overlapping classes
import csv
import numpy as np
cub_inat_mapping_path = './cub_inat_triple_exact_match.csv'
overlapping_cub_classes = []
with open(cub_inat_mapping_path, mode="r", newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        overlapping_cub_classes.append(row[3])
overlapping_cub_classes, len(overlapping_cub_classes)
#
accs = []
sup_class_acc_file = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/mohammad_class_accuracies.csv'
with open(sup_class_acc_file, mode="r", newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        if row[0] in overlapping_cub_classes:
            accs.append(float(row[1]))
print(sum(accs)/len(accs))
# %% test accs of 247 nabirds-inat overlapping classes
import csv
import numpy as np
nabirds_inat_mapping_path = './nabirds_inat_triple_exact_match.csv'
overlapping_nabirds_classes = []
with open(nabirds_inat_mapping_path, mode="r", newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        overlapping_nabirds_classes.append(int(row[3]))
overlapping_nabirds_classes, len(overlapping_nabirds_classes)
#
accs = []
sup_class_acc_file = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/nabirds/transfg/orig_mix_class_accuracies.csv'
with open(sup_class_acc_file, mode="r", newline="", encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    for row in csv_reader:
        if int(row[0]) in overlapping_nabirds_classes:
            accs.append(float(row[1]))
print(sum(accs)/len(accs))
# %%
import matplotlib.pyplot as plt

# Example lists of predicted confidences and probabilities
confidences = [0.8, 0.6, 0.7, 0.9, 0.4]
probabilities = [0.9, 0.7, 0.8, 0.85, 0.5]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(probabilities, confidences, color='blue', marker='o', alpha=0.7)
plt.xlabel('Predicted Probabilities')
plt.ylabel('Predicted Confidences')
plt.title('Confidence vs. Probability')
plt.grid(True)
plt.show()
# %%
import matplotlib.pyplot as plt

# Example lists of predicted confidences and probabilities
confidences = [0.8, 0.6, 0.7, 0.9, 0.4]
probabilities = [0.9, 0.7, 0.8, 0.85, 0.5]

# Create a histogram
plt.figure(figsize=(8, 6))
plt.hist(confidences, bins=10, color='blue', alpha=1, label='Confidences')
# plt.hist(probabilities, bins=10, color='orange', alpha=1, label='Probabilities')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Confidences and Probabilities')
plt.legend()
plt.grid(True)
plt.show()
# %%
import matplotlib.pyplot as plt

# Example lists of predicted confidences and probabilities
confidences = [0.7, 0.6, 0.8, 0.5, 0.4]
probabilities = [0.8, 0.7, 0.9, 0.6, 0.5]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(probabilities, confidences, color='blue', marker='o', alpha=0.7)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlabel('Predicted Probabilities')
plt.ylabel('Predicted Confidences')
plt.title('Under-Confident Model')
plt.grid(True)
plt.show()

# %% test with flybird_cub_test
import os
import shutil
path = '/home/tin/datasets/flybird_nabirds_test/'
orig_path = '/home/tin/datasets/nabirds/test/'
label_folders = os.listdir(path)
none_num=0
for folder in label_folders:
    folder_path = f"{path}/{folder}"
    orig_folder_path = f"{orig_path}/{folder}"
    if len(os.listdir(folder_path)) == 0:
        none_num +=1
        firstfile = os.listdir(orig_folder_path)[0]
        shutil.copy(f"{orig_folder_path}/{firstfile}", folder_path)
none_num
# %% example of Mann Whitney U testing
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate two sets of sample data
sample_a = np.random.normal(5, 2, 50)
sample_b = np.random.normal(7, 2, 50)
print(sample_a)
# Perform the Wilcoxon-Mann-Whitney test
statistic, p_value = mannwhitneyu(sample_a, sample_b)

# Display the results
print("Mann-Whitney U statistic:", statistic)
print("P-value:", p_value)
# Create a box plot to visualize the data
plt.boxplot([sample_a, sample_b], labels=["Sample A", "Sample B"])
plt.title("Box Plot of Sample Data")
plt.ylabel("Value")
plt.show()

# %% detect bird and remove images
import os
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, json

from tqdm import tqdm

# Load Mask2Former to detect bird
device = 'cuda:6'
mask2former_image_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
mask2former_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
mask2former_model.to(device)

def get_mask_one_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = mask2former_image_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = mask2former_model(**inputs)

    # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # Perform post-processing to get semantic segmentation map
    pred_semantic_map = mask2former_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    np_img = pred_semantic_map.detach().cpu().numpy()

    return np_img

def get_mask_object_bbox(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    y, y_end = np.where(rows)[0][[0, -1]]
    x, x_end = np.where(cols)[0][[0, -1]]
    width = x_end - x + 1
    height = y_end - y + 1
    return x, y, width, height

path = '/home/tin/datasets/nabirds/gen_data/inpaint_images/test_inpaint_2/'
from tqdm import tqdm
label_folders = os.listdir(path)
removed_list = []
for label_f in tqdm(label_folders):
    img_files = os.listdir(f"{path}/{label_f}")
    img_paths = [f"{path}/{label_f}/{p}" for p in img_files]
    # if detect path
    sub_removed_list = []
    for p in img_paths:
        if '.DS_Store' in p:
            continue
        image = cv2.imread(p)
        try:
            mask = get_mask_one_image(p)
        except:
            print(image_path)
        mask = mask == 14 # get the mask of the bird
        all_zero = np.all(mask == 0)

        if not all_zero:
            removed_list.append(p)
            sub_removed_list.append(p)
            print('haha')
    print(len(sub_removed_list))

print(len(removed_list))
# %%
print(len(removed_list))
# %%
for p in removed_list:
    os.remove(p)
# %% --remove sky images from test_inpaints
test_inpaint_path = '/home/tin/datasets/nabirds/gen_data/big_bb_on_birds_non_flybird_test/big_bb_on_birds_test/'
flybird_path = '/home/tin/datasets/nabirds/flybird_nabirds_test/'

from tqdm import tqdm

label_folders = os.listdir(flybird_path)
for label_f in tqdm(label_folders):
    flybird_img_files = os.listdir(f"{flybird_path}/{label_f}")
    for img_f in flybird_img_files:
        inpaint_img_p = f"{test_inpaint_path}/{label_f}/{img_f}"
        os.remove(inpaint_img_p)
    

# %% cat images
import cv2
import numpy as np
import os

# Function to concatenate images in a folder
def concatenate_images(folder_path, output_path, images_per_row=5):
    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Sort the image files to maintain order
    image_files.sort()

    # Initialize variables
    images = []
    row_images = []
    row_count = 0

    max_width = 224*images_per_row
    # Loop through the image files
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))

        
        row_images.append(image)
        row_count += 1

        if row_count == images_per_row:
            # Concatenate images horizontally
            concatenated_row = np.concatenate(row_images, axis=1)
            images.append(concatenated_row)
            row_images = []
            row_count = 0

    # Concatenate any remaining images
    if row_images:
        concatenated_row = np.concatenate(row_images, axis=1)
        last_row = np.zeros((concatenated_row.shape[0], max_width, 3))
        last_row[:, :concatenated_row.shape[1], :] = concatenated_row
        images.append(last_row)

    # Concatenate rows vertically to create the final image
    final_image = np.concatenate(images, axis=0)

    # Save the final concatenated image
    cv2.imwrite(output_path, final_image)


# folder_path = "/home/tin/datasets/nabirds/gen_data/inpaint_images/test_inpaint/0299/"
folder_path = "/home/tin/datasets/nabirds/gen_data/augsame_images_small_diff_30_added_3/0299"
output_path = "output_image_augsame_3.jpg" 
images_per_row = 5          

concatenate_images(folder_path, output_path, images_per_row)
print("Images concatenated successfully!")

# %%
irr_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/60_BIRD_ORIG_IRRELEVANT_cub_single_mohammad_08_20_2023-23:34:13/augirrelevant_missclassification.txt'
same_path = '/home/tin/projects/reasoning/cnn_habitat_reaasoning/results/cub/mohammad/SAME_cub_single_mohammad_08_16_2023-00:32:08/augsame_missclassification.txt'
habitat_diff_path = '/home/tin/projects/reasoning/plain_clip/bird_pairs_visually_same_habitat_diff.txt'

with open(irr_path, 'r') as file:
    lines = file.readlines()[1:]

irr_data = []
for line in lines:
    elements = line.strip().split('*')
    if len(elements) >= 3:
        irr_data.append([elements[0].replace('_', ' '), elements[1].replace('_', ' '), elements[2]])
irr_data[:10]
# %%
with open(same_path, 'r') as file:
    lines = file.readlines()[1:]

same_data = []
for line in lines:
    elements = line.strip().split('*')
    if len(elements) >= 3:
        same_data.append([elements[0].replace('_', ' '), elements[1].replace('_', ' '), elements[2]])
same_data[:10]
# %%
# check if how many classes augsame helps
irr_same_list = []
irr_same_list_0 = []
for irr_triplet in irr_data:
    irr_pair = irr_triplet[:-1]
    irr_pair1 = [irr_pair[0][4:], irr_pair[1][4:]]
    irr_pair2 = [irr_pair[1][4:], irr_pair[0][4:]]
    for same_triplet in same_data:
        same_pair = same_triplet[:-1]
        same_pair1 = [same_pair[0][4:], same_pair[1][4:]]
        same_pair2 = [same_pair[1][4:], same_pair[0][4:]]
        if irr_pair1 == same_pair1 or irr_pair1 == same_pair2 or irr_pair2 == same_pair1 or irr_pair2 == same_pair2:
            if irr_triplet[-1] < same_triplet[-1]:
                if irr_triplet in irr_same_list_0:
                    continue
                irr_same_list_0.append(irr_triplet)
                irr_same_list.append([irr_triplet, same_triplet[-1]])
len(irr_same_list_0)
# %%
irr_same_list_0
# %%
with open(habitat_diff_path, 'r') as file:
    lines = file.readlines()
habitat_diff_data = []
for line in lines:
    elements = line.strip().split(' - ')
    if len(elements) >= 2:
        habitat_diff_data.append([elements[0].replace('-', ' '), elements[1].replace('-', ' ')])

# %%
irr_same_list = []
for irr_triplet in irr_data:
    irr_pair = irr_triplet[:-1]
    irr_pair1 = [irr_pair[0][4:], irr_pair[1][4:]]
    irr_pair2 = [irr_pair[1][4:], irr_pair[0][4:]]
    if irr_pair1 in habitat_diff_data or irr_pair2 in habitat_diff_data:
        # check augsame
        for same_triplet in same_data:
            same_pair = same_triplet[:-1]
            same_pair1 = [same_pair[0][4:], same_pair[1][4:]]
            same_pair2 = [same_pair[1][4:], same_pair[0][4:]]
            if irr_pair1 == same_pair1 or irr_pair1 == same_pair2 or irr_pair2 == same_pair1 or irr_pair2 == same_pair2:
                irr_same_list.append([irr_triplet, same_triplet[-1]])
                
# %%
irr_same_list
# %%
num_improve = 0
data = []
for irr_same in irr_same_list:
    irr_list = irr_same[0]
    irr_count = int(irr_list[-1])
    same_count = int(irr_same[1])
    if irr_count > same_count:
        data.append(irr_same)
        num_improve+=1
# %%
num_improve
# %%
import csv
# Define the output file name and column names
output_file = 'irrelevant_same_diff_habitat.csv'
column_names = ['bird1', 'bird2', 'irr_missed', 'same_missed']

# Open the CSV file for writing
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row with column names
    writer.writerow(column_names)

    # Write the data rows
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            writer.writerow([item[0][0], item[0][1], item[0][2], item[1]])
# %%
