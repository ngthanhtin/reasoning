# %%
import json

path = "/home/tin/projects/reasoning/plain_clip/descriptors/inaturalist2021/425_sachit_descriptors_inaturalist.json"
id_path = "/home/tin/projects/reasoning/plain_clip/descriptors/inaturalist2021/replaced_425_ID_descriptors_inaturalist.json"
f = open(path, 'r')
data = json.load(f)
f = open(id_path, 'r')
id_data = json.load(f)
data, len(data), id_data, len(id_data)

# %%
new_data = {}
for k in data:
    # if '(' not in k:
    #     new_data[k] = data[k]
    #     new_data[k] += id_data[k]
    new_data[k] = data[k]
    new_data[k] += id_data[k]

len(new_data), new_data.keys()

# %%
json_object = json.dumps(new_data, indent=4)
with open("/home/tin/projects/reasoning/plain_clip/descriptors/inaturalist2021/replaced_425_additional_sachit_descriptors_inaturalist.json", "w") as outfile:
    outfile.write(json_object)
# %%
f1 = open("correct_nabirds_id_paths.txt", 'r')
f2 = open("incorrect_nabirds_id2_paths.txt", 'r')

lines1 = f1.readlines()
lines1 = [line[:-1] for line in lines1]
lines2 = f2.readlines()
lines2 = [line[:-1] for line in lines2]
lines1
# %%
paths = []
for line1 in lines1:
    if line1 in lines2: 
        paths.append(line1)

paths, len(paths)
# %% save paths
f3 = open("abcd_nabirds.txt", "w")
for item in paths:
    print('haha')
    f3.write(str(item) + '\n')

# %% -- test inaturalist2021---
import json
desc_path = 'descriptors/inaturalist2021/chatgpt_descriptors_inaturalist.json'
f = open(desc_path, 'r')
data = json.load(f)
len(data), data.keys()
# %%
allaboutbirds_birdname2sciname_path = "/home/tin/projects/reasoning/scraping/birdname_2_sciname_allaboutbirds.json"
f = open(allaboutbirds_birdname2sciname_path, 'r')
allaboutbirds_birdname2sciname_dict = json.load(f)
allaboutbirds_sciname2birdname_dict = {v:k for k,v in allaboutbirds_birdname2sciname_dict.items()}
print(allaboutbirds_sciname2birdname_dict)

num_overlapped = 0
overlapped_dict = {}
for inat_sci_name in data.keys():
    if inat_sci_name in allaboutbirds_sciname2birdname_dict.keys():
        num_overlapped += 1
        overlapped_dict[inat_sci_name] = allaboutbirds_sciname2birdname_dict[inat_sci_name]
num_overlapped

json_objects = json.dumps(overlapped_dict, indent=4)
with open("overlapped_inat_allaboutbirds.json", 'w') as f:
    f.write(json_objects)


# %%-- create inat-alternative descriptions
original_inat_desc_path = '/home/tin/projects/reasoning/plain_clip/descriptors/inaturalist2021/sachit_descriptors_inaturalist.json'
f = open(original_inat_desc_path, 'r')
original_inat_desc = json.load(f)

original_inat_desc.keys()
# %%
new_inat_desc_path = '/home/tin/projects/reasoning/plain_clip/descriptors/inaturalist2021/425_sachit_descriptors_inaturalist.json'
new_dict = {}
for key in overlapped_dict:
    new_dict[key] = original_inat_desc[key]

# save json
json_objects = json.dumps(new_dict, indent=4)
with open(new_inat_desc_path, 'w') as f:
    f.write(json_objects)

# %% --- replace all sci names to bird names
import json

original_inat_desc_path = '/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json'
f = open(original_inat_desc_path, 'r')
inat_desc = json.load(f)

inat_desc.keys()

# %%
overlapped_path = "/home/tin/reasoning/plain_clip/overlapped_inat_allaboutbirds.json"
f = open(overlapped_path, 'r')
overlapped_dict = json.load(f)

overlapped_dict.values()
# %%
for k, v in overlapped_dict.items():
    overlapped_dict[k] = v.replace('_', ' ')
overlapped_dict.values()

# %%
for k, vs in inat_desc.items():
    sci_name = k
    bird_name = overlapped_dict[k]
    inat_desc[k][-1] = vs[-1].replace(bird_name, sci_name)
    inat_desc[k][-2] = vs[-2].replace(bird_name, sci_name)
    inat_desc[k][-3] = vs[-3].replace(bird_name, sci_name)
    inat_desc[k][-4] = vs[-4].replace(bird_name, sci_name)

# %%
# save json
new_inat_desc_path = '/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/replaced_425_ID_descriptors_inaturalist.json'
json_objects = json.dumps(inat_desc, indent=4)
with open(new_inat_desc_path, 'w') as f:
    f.write(json_objects)
# %% Create ID2 replaced
for k, v in inat_desc.items():
    inat_desc[k] = v[:-1]

new_inat_desc_path = '/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/replaced_425_ID2_descriptors_inaturalist.json'
json_objects = json.dumps(inat_desc, indent=4)
with open(new_inat_desc_path, 'w') as f:
    f.write(json_objects)



# %% fix classname to idex.classname
import os
from shutil import move

path_to_fix = '/home/tin/projects/reasoning/plain_clip/retrieval_cub_images_by_texts_noinpaint_unsplash/'
path_ref = '/home/tin/datasets/cub/CUB/images/'

classname_idx = {}

list_class_fix = os.listdir(path_to_fix)
list_class_ref = os.listdir(path_ref)

for cls in list_class_ref:
    idx, classname = cls.split('.')
    # idx = int(idx)
    classname = classname.replace('_', ' ')
    split_key = classname.split(' ')
    if len(split_key) > 2: 
        classname = '-'.join(split_key[:-1]) + " " + split_key[-1]
    classname_idx[classname] = idx

for cls in list_class_fix:
    idx = classname_idx[cls]
    
    orig_path = f'{path_to_fix}/{cls}'
    cls = cls.replace(' ', '_')
    changed_path = f'{path_to_fix}/{idx}.{cls}'

    move(orig_path, changed_path)


# %%
# %%
import os
import cv2
import numpy as np
import math
import pandas as pd
import json

def merge_images(image_paths, class_name="EMPTY"):
    """
    image_paths: paths of images that need to be concatenated
    class_name: class of input images
    return np image
    """

    images = []
    image_names = []
    H, W = 224, 224
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (W, H))
        images.append(image)
            
        # get file name
        image_name = image_path.split('/')[-1]
        # TODO: Append image name below the image

    len_images = len(images)
    columns, rows = 5, math.ceil(len_images/5) # predefined number of columns is 5

    extended_images = []
    for i in range(rows):
        row_images = []
        for j in range(columns):
            mapped_idx = 5*i + j
            if mapped_idx < len_images:
                row_images.append(images[mapped_idx])
            else:
                row_images.append(np.zeros((H, W, 3)))
        
        row_images = np.array(row_images)
        row_image = np.concatenate(row_images, axis=1)
        extended_images.append(row_image)
    
    extended_images = np.array(extended_images)
    extended_image = np.concatenate(extended_images, axis=0)
    
    return extended_image

def merge_dataset():
    
    data_path = './retrieval_cub_images_by_texts_noinpaint_unsplash_query/'
    save_img_folder = './merged_cub_retrieval/'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)

    img_folders = os.listdir(data_path)

    for folder in img_folders:
        img_paths = os.listdir(f"{data_path}/{folder}")
        img_paths = [f'{data_path}/{folder}/{p}' for p in img_paths if 'txt' not in p]
        # run to merge images
        merged_img = merge_images(img_paths, folder)
        cv2.imwrite(f"{save_img_folder}/{folder}.jpg", merged_img)
# %%
merge_dataset()
# %%
import wikipedia as wiki
import pandas as pd
import json

desc_path = 'descriptors/inaturalist2021/chatgpt_descriptors_inaturalist.json'
f = open(desc_path, 'r')
data = json.load(f)
sci_names = list(data.keys())
sci_names[:10]
# %%

comm_names = []
for sci_name in sci_names[:5]:
    search_out=wiki.search(sci_name,results=1)
    comm_names.append(search_out[0])

comm_names
# %%



# %%
import json

def find_duplicates(input_list):
    # Create an empty dictionary to store the count of each element
    element_count = {}

    # Iterate through the list and count the occurrences of each element
    for element in input_list:
        element_count[element] = element_count.get(element, 0) + 1

    # Create a list to store the duplicate elements
    duplicates = []

    # Check if the count of any element is greater than 1 (i.e., it's a duplicate)
    for element, count in element_count.items():
        if count > 1:
            duplicates.append(element)

    return duplicates

path = "sci2comm_inat_full.json"
f = open(path, 'r')
data = json.load(f)
value2key = {}
for k,v in data.items():
    if v not in value2key:
        value2key[v] = [k]
    else:
        value2key[v].append(k)

result = find_duplicates(data.values())
print("Duplicates in the list:", result)
print(len(result))


# %%
# make multi label for CUB dataset
import pandas as pd
import numpy as np
import json

cub_path = '/home/tin/datasets/cub/CUB/images/'
n_clusters = 200
class_cub_cluster_path = f'class_cub_clusters_{n_clusters}.json'
# %%
f = open(class_cub_cluster_path, 'r')
cluster_data = json.load(f)

# %% graph cluster
folderclasses = os.listdir(cub_path)
folderclass2class = {}
graph = {}
labelname2labelidx = {}
for cls in folderclasses:
    name = cls.split('.')[1]
    label_idx = int(cls.split('.')[0])

    if len(name.split('_')) > 2:
        name_parts = name.split('_')
        if len(name.split('_')) == 3:
            name = name_parts[0] + '-' + name_parts[1] + ' ' + name_parts[2]
        else:
            name = name_parts[0] + '-' + name_parts[1] + '-' + name_parts[2] + ' ' + name_parts[3]
    else:
        name = name.replace('_', ' ')

    folderclass2class[cls] = name

    labelname2labelidx[name] = label_idx

for k,v in cluster_data.items():
    for label in v:
        label_idx = labelname2labelidx[label]
        if label_idx not in graph:
            graph[label_idx] = []
        for label in v:
            vertice = labelname2labelidx[label]
            graph[label_idx].append(vertice)
graph, len(graph)
# %% save graph
file_path = f"cub_multilabel_{n_clusters}.json"
with open(file_path, "w") as json_file:
    json.dump(graph, json_file, indent=4)

# %% -- make black border for images
import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/tin/projects/reasoning/plain_clip/Painted_Bunting_0102_16642.jpg')

# Define patch size
patch_size = 24

# Get image dimensions
image = cv2.resize(image, (576, 576))
height, width = image.shape[:2]
cv2.imwrite('resize_image.png', image)
#%%
# Calculate the number of patches in rows and columns
num_rows = height // patch_size
num_cols = width // patch_size

# Initialize an empty canvas for the final image
final_image = np.zeros((height + 2*24, width+2*24, 3), dtype=np.uint8)

# Loop through each patch
for row in range(num_rows):
    for col in range(num_cols):
        # Calculate patch boundaries
        y_start = row * patch_size
        y_end = y_start + patch_size
        x_start = col * patch_size
        x_end = x_start + patch_size
        
        # Extract the patch
        patch = image[y_start:y_end, x_start:x_end]
        
        # Add a black border to the patch
        bordered_patch = cv2.copyMakeBorder(patch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Calculate position for the patch in the final image
        y_pos = row * (patch_size + 2)
        x_pos = col * (patch_size + 2)
        
        # Place the bordered patch on the final image
        print(y_pos, y_pos+patch_size+2) 
        print(x_pos, x_pos+patch_size+2)
        # final_image[y_pos:y_pos+patch_size+2, x_pos:x_pos+patch_size+2] = bordered_patch
        final_image[y_pos:y_pos+patch_size+2, x_pos:x_pos+patch_size+2] = bordered_patch

# Show and save the final image
cv2.imwrite('final_image_with_borders.png', final_image)
# %%
# convert nabirds name
import pandas as pd
import cv2

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
labelname2labelidx
# %% read imagenet labels
from torchvision.datasets import ImageFolder
imagenet_dataset = ImageFolder('/home/tin/datasets/imagenet/train/', transform=None)
imagenet_class_to_idx = imagenet_dataset.class_to_idx
imagenet_idx_to_class = {v:k for k,v in imagenet_class_to_idx.items()}

# %% read pet labels
import os, json
images_path = '/home/tin/datasets/oxford_pet/train/'
folder_labels = os.listdir(images_path)
folder_to_lowercase_dict =  {k: k.lower() for k in folder_labels}
lowercase_to_folder_dict =  {k.lower(): k for k in folder_labels}

# %% read inat folder
import os, json
with open(os.path.join('/home/tin/datasets/inaturalist2021_onlybird/bird_classes.json'), 'r') as f:
    class_meta = json.load(f)
    # easy mapping for class id, class name and corresponding folder name
    
idx2class = {int(id): name for id, name in class_meta['name'].items()} # if use subset
class2idx = {v: k for k, v in idx2class.items()}

cls_id2folder_name = {int(id): name for id, name in class_meta['image_dir_name'].items()} # if use subset
folder_name2cls_id = {v: k for k, v in cls_id2folder_name.items()}

inat_class2folder_name = {cls: cls_id2folder_name[id] for cls, id in class2idx.items()}
inat_class2folder_name, len(inat_class2folder_name)
# %% Cretate allaboutbirds_example_images_path = 'cub_allaboutbirds_example_images.json'
import json, os
from collections import OrderedDict

dataset = 'nabirds' # cub
example_images = OrderedDict()
text_description_path = '/home/tin/projects/reasoning/plain_clip/descriptors/nabirds/no_ann_additional_chatgpt_descriptors_nabirds.json'
dataset_descs = open(text_description_path, 'r')
descs = json.load(dataset_descs)

num_examples = 50
# images_path = '/home/tin/datasets/nabirds/train/'
# images_path = '/home/tin/datasets/imagenet/train/'
# images_path = '/home/tin/datasets/oxford_pet/train/'
images_path = '/home/tin/datasets/nabirds/gen_data/onlybird_images_train/'

for i, k in enumerate(descs.keys()):
    example_images[k] = []
    # convert common name to classs label
    if dataset == 'cub':
        k_ = k.replace('-','_')
        k_ = k_.replace(' ','_')
        i_ = str(i+1)
        if len(i_) == 1:
            i_ = '00'+i_
        elif len(i_) == 2:
            i_ = '0'+i_
    elif dataset == 'nabirds':
        folder_label = str(labelname2labelidx[k])
        folder_label = '0'*(4-len(folder_label)) + folder_label
    elif dataset == 'imagenet':
        folder_label = imagenet_idx_to_class[i]
    elif dataset == 'pet':
        folder_label = lowercase_to_folder_dict[k.lower()]
    elif dataset == 'inaturalist':
        folder_label = inat_class2folder_name[k]

    img_files = os.listdir(f"{images_path}/{folder_label}")
    imgs_files = img_files[:num_examples]
    imgs_files = [f"{images_path}/{folder_label}/{p}" for p in imgs_files]
    example_images[k] = imgs_files
# %%
example_images
# %%
with open(f"./image_descriptions/{dataset}/{dataset}_only_birds_example_images_{num_examples}.json", "w") as outfile:
    json.dump(example_images, outfile)
# %% test bird visually similar but habitat different
from itertools import permutations
import json

visual_path = '/home/tin/projects/reasoning/plain_clip/shape_size_color_cub_clusters_49.json'
visual_groups = open(visual_path, 'r')
visual_groups = json.load(visual_groups)
habitat_path = '/home/tin/projects/reasoning/plain_clip/class_cub_clusters_50.json'
habitat_groups = open(habitat_path, 'r')
habitat_groups = json.load(habitat_groups)

bird_pairs = [] # pairs in which visually similar but different habitat
for i, vg in visual_groups.items():
    if len(vg) == 1:
        continue
    permutation_pairs = set(permutations(vg, 2))
    for pair in permutation_pairs:
        is_same_habitat = False
        for hg in habitat_groups.items():
            if pair[0] in hg and pair[1] in hg:
                is_same_habitat = True
                break
        if not is_same_habitat:
            bird_pairs.append(pair)
# %%
unique_bird_pairs = set(frozenset(pair) for pair in bird_pairs)
unique_bird_pairs = [tuple(fs) for fs in unique_bird_pairs]
len(unique_bird_pairs)
# %%
# Define the filename
filename = "bird_pairs_visually_same_habitat_diff.txt"
# Open the file for writing
with open(filename, "w") as file:
    for pair in unique_bird_pairs:
        file.write(f"{pair[0]} - {pair[1]}\n")
# %%
unique_birds = set()
for p in unique_bird_pairs:
    unique_birds.add(p[0])
    unique_birds.add(p[1])
len(unique_birds)

# %% Create test, train set for Oxford pet
pet_path = '/home/tin/datasets/oxford_pet/images/'
trainval_anno_path = '/home/tin/datasets/oxford_pet/annotations/trainval.txt'
test_anno_path = '/home/tin/datasets/oxford_pet/annotations/test.txt'

# %%
image_ids = []
labels = []
with open(test_anno_path) as file:
    for line in file:
        image_id, label, *_ = line.strip().split()
        image_ids.append(image_id)
        labels.append(int(label)-1)
# %%
classes = [
    " ".join(part.title() for part in raw_cls.split("_"))
    for raw_cls, _ in sorted(
        {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, labels)},
        key=lambda image_id_and_label: image_id_and_label[1],
    )
    ]

idx_to_class = dict(zip(range(len(classes)), classes))
idx_to_class

# %%
save_path = '/home/tin/datasets/oxford_pet/test/'

import os 
import shutil
for img_id, label in zip(image_ids, labels):
    img_path = f'{pet_path}/{img_id}.jpg'

    class_name = idx_to_class[label]
    class_name = class_name.replace("_", " ")
    if not os.path.exists(f"{save_path}/{class_name}"):
        os.makedirs(f"{save_path}/{class_name}")
    shutil.copy(img_path, f"{save_path}/{class_name}")
    
# %%
print(len(image_ids))
# %%
import pandas as pd
dataset = pd.read_csv("/home/tin/datasets/oxford_pet/annotations/data.txt", sep=" ")
dataset
# %%
dataset['nID'] = dataset['ID'] - 1
decode_map = idx_to_class
def decode_label(label):
    return decode_map[int(label)]
dataset["class"] = dataset["nID"].apply(lambda x: decode_label(x))

dataset

# %% test openai API
import os
import openai
openai.organization = "YOUR_ORG_ID"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()

# %%
import torch
similarity_matrix_chunk = torch.rand((64, 98))
# Assuming similarity_matrix_chunk is your PyTorch tensor
similarity_matrix_chunk[similarity_matrix_chunk < 0.2] = 0

# Remove the zeros within each row
nonzero_rows = torch.nonzero(similarity_matrix_chunk)

values = similarity_matrix_chunk[nonzero_rows[:, 0], nonzero_rows[:, 1]]

# Calculate the mean for each row
row_sums = torch.zeros(similarity_matrix_chunk.size(0), device=similarity_matrix_chunk.device)
row_counts = torch.zeros(similarity_matrix_chunk.size(0), device=similarity_matrix_chunk.device)

# Sum values and count non-zero elements for each row
row_sums.index_add_(0, nonzero_rows[:, 0], values)
row_counts.index_add_(0, nonzero_rows[:, 0], torch.ones_like(values))

# Calculate row means, avoiding division by zero
row_means = row_sums / row_counts.float().clamp(min=1)
print(row_means)
# row_means now contains the mean for each row

# %%
# Test max num descs
path = 'descriptors/cub/gpt_4_sachit_descriptors_cub.json'

import json
f = open(path, 'r')
desc = json.load(f)
n = [0 for i in range(13)]
for k,v in desc.items():
    # if n < len(v):
    #     n = len(v)
    n[len(v)] += 1
n
# %% create additional_sachit_descriptions_nabirds.json
import json
path = 'descriptors/nabirds/sachit_descriptors_nabirds.json'
save_path = 'descriptors/nabirds/additional_sachit_descriptors_nabirds.json'

id_path = 'descriptors/nabirds/ID_descriptors_nabirds.json'

f = open(path, 'r')
sachit_data = json.load(f)

f = open(id_path, 'r')
id_data = json.load(f)

for k, v in sachit_data.items():
    shape_desc, size_desc, habitat_desc = id_data[k][0], id_data[k][1], id_data[k][3]
    sachit_data[k].append(shape_desc)
    sachit_data[k].append(size_desc)
    sachit_data[k].append(habitat_desc)

json_objects = json.dumps(sachit_data, indent=4)
with open(save_path, 'w') as f:
    f.write(json_objects)
# %%
import cv2
import os
import numpy as np
# Define the paths to the two image folders
folder2 = "/home/tin/projects/reasoning/plain_clip/correct_cub_id_figs"
folder1 = "correct_cub_no_habitat_figs"

for filename1 in os.listdir(folder1):
    if filename1.endswith(".jpg"):
        filepath1 = os.path.join(folder1, filename1)

        for filename2 in os.listdir(folder2):
            if filename2.endswith(".jpg"):
                filepath2 = os.path.join(folder2, filename2)

                if filename2 == filename1:
                    print(f"Images {filename1} and {filename2} are identical.")
        


                    image1 = cv2.imread(filepath1)
                    image2 = cv2.imread(filepath2)

                    # Make sure both images have the same width
                    if image1.shape[1] != image2.shape[1]:
                        raise ValueError("Images must have the same width")

                    # Vertically concatenate the two images
                    concatenated_image = np.vstack((image1, image2))

                    # Save the concatenated image
                    cv2.imwrite(f'concat_images/{filename1}.jpg', concatenated_image)

                    break

# %%
import json, random
full_examples = '/home/tin/projects/reasoning/plain_clip/image_descriptions/nabirds/nabirds_example_images_50.json'
onlybird_examples = "./image_descriptions/nabirds/nabirds_only_birds_example_images_50.json"

onlybird_folder_path = '/home/tin/datasets/nabirds/gen_data/onlybird_images_train/'
# read json
with open(full_examples, 'r') as file:
    data = json.load(file)

onlybird_data = {}
for c, ps in data.items():
    for p in ps:
        label, img_name = p.split('/')[-2], p.split('/')[-1]
        onlybird_image_path = f"{onlybird_folder_path}/{label}/{img_name}"
        if c not in onlybird_data:
            onlybird_data[c] = [onlybird_image_path]
        else:
            onlybird_data[c].append(onlybird_image_path)
with open(onlybird_examples, 'w') as file:
    json.dump(onlybird_data, file, indent=4) 
# %%
import json, random

nabirds_examples_file = "./image_descriptions/nabirds/nabirds_only_birds_example_images_50.json"
nabirds_irrelevant_image_path = '/home/tin/datasets/nabirds/gen_data/augsame_images/'
nabirds_irrelevant_examples_file = './image_descriptions/nabirds/nabirds_same_example_images_50.json'
# read json
with open(nabirds_examples_file, 'r') as file:
    data = json.load(file)
new_data = {}
for c, ps in data.items():
    for p in ps:
        label, img_name = p.split('/')[-2], p.split('/')[-1]
        irrelevant_image_folder = f"{nabirds_irrelevant_image_path}/{label}"
        irrelevant_images = os.listdir(irrelevant_image_folder)

        list_irrelevant_image_path = []
        for irr_p in irrelevant_images:
            if img_name[:-4] + '_' in irr_p:
                list_irrelevant_image_path.append(f"{nabirds_irrelevant_image_path}/{label}/{irr_p}")
        
        irrelevant_image_path = random.choice(list_irrelevant_image_path)
        if c not in new_data:
            new_data[c] = list_irrelevant_image_path[:1] #[irrelevant_image_path]
        else:
            # new_data[c].append(irrelevant_image_path)
            new_data[c] += list_irrelevant_image_path[:1]
with open(nabirds_irrelevant_examples_file, 'w') as file:
    json.dump(new_data, file, indent=4) 

# %% create hierachy description
import json
desc_path = 'descriptors/cub/additional_sachit_descriptors_cub.json'
# read json
with open(desc_path, 'r') as file:
    data = json.load(file)
new_data = {}

for k, v in data.items():
    habitat = v[-1]
    habitat_1 = habitat.split('.')[0]
    habitat_2_list = habitat.split('.')[1:]
    # habitat_3_list = habitat.split('.')[2:]

    # if habitat_2_list:
    #     habitat_2 = 'habitat:'
    #     for h in habitat_2_list:
    #         habitat_2 += h + '.'
    #     if habitat_2[-2] == '.':
    #         habitat_2 = habitat_2[:-1]
    #     habitat_2 = habitat_2[:-1]

    #     new_data[k+'_A'] = v[:-1] + [habitat_1]
    #     new_data[k+'_B'] = v[:-1] + [habitat_2]
    # else:
    #     new_data[k] = v

    if habitat_2_list:
        habitat_2 = 'habitat:'
        for h in habitat_2_list:
            habitat_2 += h + '.'
        if habitat_2[-2] == '.':
            habitat_2 = habitat_2[:-1]
        habitat_2 = habitat_2[:-1]

        new_data[k+'_A'] = v[:-1] + [habitat_1]
        new_data[k+'_B'] = v[:-1] + [habitat_2]
    else:
        new_data[k] = v

with open('descriptors/hier_additional_sachit_descriptors_cub.json', 'w') as file:
    json.dump(new_data, file, indent=4) 

# %% convert black background images to blue-sky background images
path = '/home/tin/datasets/cub/CUB_no_bg_test/'
blue_sky_bg_img = '/home/tin/datasets/cub/CUB_inpaint_all_test/023.Brandt_Cormorant/Brandt_Cormorant_0001_23398.jpg'

import os
folders = os.listdir(path)

for f in folders:
    folder_image_path = f"{path}/{f}"
    os.mkdir(f"abc/{f}")

    image_path = os.listdir(folder_image_path)
    for image in image_path:
        image_full_path = f"{folder_image_path}/{image}"

        

# %%
def read_accuracies_from_file(file_path):
    with open(file_path, 'r') as f:
        # Read lines and convert each line to a float representing the accuracy
        accuracies = [float(line.strip().rstrip('%')) for line in f.readlines()]
    return accuracies

# Read accuracies from both files
# CLIP
nohabitat_accuracies_file1 = read_accuracies_from_file('class_accuracies/cub/B_32_nohabitat_class_accuracies.txt')
nohabitat_accuracies_file2 = read_accuracies_from_file('class_accuracies/cub/B_16_nohabitat_class_accuracies.txt')
nohabitat_accuracies_file3 = read_accuracies_from_file('class_accuracies/cub/L_14_nohabitat_class_accuracies.txt')
sum1 = [a + b + c for a, b, c in zip(nohabitat_accuracies_file1, nohabitat_accuracies_file2, nohabitat_accuracies_file3)]
habitat_accuracies_file1 = read_accuracies_from_file('class_accuracies/cub/B_32_habitat_class_accuracies.txt')
habitat_accuracies_file2 = read_accuracies_from_file('class_accuracies/cub/B_16_habitat_class_accuracies.txt')
habitat_accuracies_file3 = read_accuracies_from_file('class_accuracies/cub/L_14_habitat_class_accuracies.txt')
sum2 = [a + b + c for a, b, c in zip(habitat_accuracies_file1, habitat_accuracies_file2, habitat_accuracies_file3)]
# Compute the differences
differences = [(a - b)/3 for a, b in zip(sum2, sum1)]

# Pair each difference with its class index
indexed_differences = list(enumerate(differences))

# Sort the differences to find the top 10
top_10_differences_clip = sorted(indexed_differences, key=lambda x: x[1], reverse=True)[:20]

# Print out the results
print("Top 10 classes with the highest differences:")
for index, diff in top_10_differences_clip:
    print(f"Class {index}: {diff:.2f}%")

# Count non-negative differences
positive_count = sum(1 for index, diff in top_10_differences_clip if diff > 0)
zero_count = sum(1 for index, diff in top_10_differences_clip if diff == 0)

print(f"Number of positive differences: {positive_count}")
print(f"Number of zero differences: {zero_count}")

# Optionally, if you want to save these results to a file:
# with open('top_10_differences.txt', 'w') as f:
#     for index, diff in top_10_differences:
#         f.write(f"Class {index}: {diff:.2f}%\n")


# %%
def read_accuracies_from_file(file_path):
    with open(file_path, 'r') as f:
        # Read lines and convert each line to a float representing the accuracy
        accuracies = [float(line.strip().rstrip('%')) for line in f.readlines()]
    return accuracies

# Read accuracies from both files
# Unimodal
dataset = 'cub'
nohabitat_accuracies_file1 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/noaug_cnn_class_accuracy.txt')
nohabitat_accuracies_file2 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/noaug_transfg_class_accuracy.txt')
sum1 = [a + b for a, b in zip(nohabitat_accuracies_file1, nohabitat_accuracies_file2)]
same_habitat_accuracies_file1 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/same_cnn_class_accuracy.txt')
same_habitat_accuracies_file2 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/same_transfg_class_accuracy.txt')
sum2 = [a + b for a, b in zip(same_habitat_accuracies_file1, same_habitat_accuracies_file2)]
group_habitat_accuracies_file1 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/group_cnn_class_accuracy.txt')
group_habitat_accuracies_file2 = read_accuracies_from_file(f'/home/tin/projects/reasoning/cnn_habitat_reaasoning/class_accuracies/{dataset}/group_transfg_class_accuracy.txt')
sum3 = [a + b for a, b in zip(group_habitat_accuracies_file1, group_habitat_accuracies_file2)]
# Compute the differences
differences = [(a - c + b - c)/2 for a, b, c in zip(sum3, sum2, sum1)]

# Pair each difference with its class index
indexed_differences = list(enumerate(differences))

# Sort the differences to find the top 10
top_10_differences_uni = sorted(indexed_differences, key=lambda x: x[1], reverse=True)[:20]

# Print out the results
print("Top 10 classes with the highest differences:")
for index, diff in top_10_differences_uni:
    print(f"Class {index}: {diff:.2f}%")

# Count non-negative differences
positive_count = sum(1 for index, diff in top_10_differences_uni if diff > 0)
zero_count = sum(1 for index, diff in top_10_differences_uni if diff == 0)

print(f"Number of positive differences: {positive_count}")
print(f"Number of zero differences: {zero_count}")
# %%
print(len(top_10_differences_clip))
print(len(top_10_differences_uni))
 # 101 vs 87
 # %%
 # find parent names
import json
parents = {}
taxonomy = 'cub_taxonomy.json'
f = open(taxonomy, 'r')
taxonomy_data = json.load(f)

cub_path = '/home/tin/datasets/cub/CUB/images/'
folder = os.listdir(cub_path)
folder.sort()

for i, name in enumerate(folder):
    name = name[4:].replace('_', ' ')
    
    # find parent of the class
    for tax_name in taxonomy_data:
        comm_name = tax_name['comName'].replace('-', ' ')
        if '\'s' in comm_name:
            comm_name = comm_name.replace('\'s', '')

        parent_name = tax_name['familyComName']
        if name.lower() in comm_name.lower():
            parents[i] = parent_name
            break
len(parents)
 # %%
align = 0
approx_align = 0
approx_rate = 2

description_path = f"./descriptors/cub/ID_descriptors_cub.json"

f = open(description_path, 'r')
documents = json.load(f)
names2indexes = {k:i for i, (k,v) in enumerate(documents.items())}
indexes_names = {i:k for i, (k,v) in enumerate(documents.items())}

group_path = f"class_cub_clusters_50.json"
f = open(group_path, 'r')
groups = json.load(f)

indexes = []
# 1 vs 2
for index1, diff1 in top_10_differences_clip[:20]:
    if index1 == 125:
        continue
    if diff1 <= 0:
        continue
    for index2, diff2 in top_10_differences_uni[:20]:
        if index2 == 125:
            continue
        if diff2 <= 0:
            continue
        
        for group, group_item in groups.items():
            if indexes_names[index1] in group_item and indexes_names[index2] in group_item:
                align+=1
                indexes.append([index1, index2, diff1, diff2])
print(align)
# 1 vs 1
indexes_11 = []
align = 0
for index1, diff1 in top_10_differences_clip[:20]:
    if index1 == 125:
        continue
    if diff1 <= 0:
        continue
    for index2, diff2 in top_10_differences_clip[:20]:
        if index2 == index1:
            continue
        if index2 == 125:
            continue
        if diff2 <= 0:
            continue
        
        for group, group_item in groups.items():
            if indexes_names[index1] in group_item and indexes_names[index2] in group_item:
                align+=1
                indexes_11.append([index1, index2, diff1, diff2])
print(align)
# 2 vs 2
indexes_22 = []
align = 0
for index1, diff1 in top_10_differences_uni[:20]:
    if index1 == 125:
        continue
    if diff1 <= 0:
        continue
    for index2, diff2 in top_10_differences_uni[:20]:
        if index2 == index1:
            continue
        if index2 == 125:
            continue
        if diff2 <= 0:
            continue
        
        for group, group_item in groups.items():
            if indexes_names[index1] in group_item and indexes_names[index2] in group_item:
                align+=1
                indexes_22.append([index1, index2, diff1, diff2])
print(align)
#%%
print(indexes)
print('-----')
print(indexes_11)
print('------')
print(indexes_22)
# %%
import os
cub_path = '/home/tin/datasets/cub/CUB/images/'
folder = os.listdir(cub_path)
folder.sort()

for p1, p2, d1, d2 in indexes:
    print(p1, p2)
    # print(f"{folder[p1]} vs {folder[p2]}, {d1} vs {d2}")
    print('-----------')

# %%
colors = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 
    'lime', 'pink', 'teal', 'lavender', 'maroon', 'brown', 
    'coral', 'turquoise', 'gold'
]

colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#ff1493', '#00ced1', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5'
]
# Identify the gray colors in the list
gray_colors = ['#7f7f7f', '#c7c7c7']

#1f77b4 (replacement)
#aec7e8 (replacement)

# Get additional colors from the 'tabx' palette to replace the grays
# color_map = plt.cm.get_cmap('turbo', 20)
# replacement_colors = [mcolors.rgb2hex(color_map(i)) for i in range(20) if mcolors.rgb2hex(color_map(i)) not in colors]
# print(replacement_colors)
# # Replace the gray colors with new colors from the palette
# for gray_color in gray_colors:
#     if gray_color in colors:
#         colors[colors.index(gray_color)] = replacement_colors.pop(0)  # Replace and remove the used color


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

color_map = plt.get_cmap('tab20')
colors_50 = [color_map(i % color_map.N) for i in range(50)]

# Converting color codes to hexadecimal format
hex_colors_50 = [mcolors.rgb2hex(color) for color in colors_50]

# colors = hex_colors_50

# color_codes = [
#     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
#     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
#     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
#     '#17becf', '#9edae5', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
#     '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
#     '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
#     '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#1f77b4', '#aec7e8',
#     '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
#     '#9467bd', '#c5b0d5'
# ]

# # Function to check if a color code is gray
def is_gray(color_code):
    r = color_code[1:3]
    g = color_code[3:5]
    b = color_code[5:7]
    return r == g == b

# # Checking for gray colors in the list
# gray_colors = [color for color in colors if is_gray(color)]

# colors = [color for color in colors if not is_gray(color)]
print(colors)

# %%

import matplotlib.pyplot as plt

import os
cub_path = '/home/tin/datasets/cub/CUB/images/'
folder = os.listdir(cub_path)
folder.sort()

# for p1, p2, d1, d2 in indexes:
#     print(f"{folder[p1]} vs {folder[p2]}, {d1} vs {d2}")
#     print('-----------')

names_1 = [folder[i][4:].replace('_', ' ') for i, j in top_10_differences_clip]
indexes_1 = [i for i, j in top_10_differences_clip]
names_2 = [folder[i][4:].replace('_', ' ') for i, j in top_10_differences_uni]
indexes_2 = [i for i, j in top_10_differences_uni]

ratio = 1
values_left = [21.43, 16.67, 14.45, 14.44, 14.44, 11.11, 8.70, 7.78, 7.78, 7.78, 7.78, 7.78, 7.78, 6.90, 6.67, 6.67, 6.67, 6.41, 6.06, 6.06]
values_right = [19.99, 18.34, 16.67, 16.67, 15.79, 15.01, 15.00, 14.99, 13.34, 10.87, 10.86, 10.35, 10.34, 10.00, 10.00, 8.62, 8.34, 8.34, 8.33, 6.91]
values_left = [i/ratio for i in values_left]
values_right = [i/ratio for i in values_right]

categories = [f'{i}' for i in range(len(values_right))]

matches = []

p1_in_clip_s = []
p2_in_uni_s = []

for p1, p2, d1, d2 in indexes:
    p1_in_clip = indexes_1.index(p1)
    p1_in_clip_s.append(p1_in_clip)
    p2_in_uni = indexes_2.index(p2)
    p2_in_uni_s.append(p2_in_uni)
    matches.append((p1_in_clip, p2_in_uni))
# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# initialize colors for each bar
match_colors_1 = ['gray' for i in range(20)]
match_colors_1[0] = 'gold'
match_colors_1[6] = 'gold'
match_colors_1[18] = 'gold'
match_colors_2 = ['gray' for i in range(20)]
match_colors_2[4] = 'coral'
match_colors_2[11] = 'coral'

parent_graph = {i: [] for i in range(20)}
for p1, p2 in matches:
    parent_graph[p1].append(p2)

color_index = 0
index_vs_color = {}
for index1 in range(20):
    if parent_graph[index1]:
        for index2 in parent_graph[index1]:
            if match_colors_2[index2] != 'gray':
                match_colors_1[index1] = match_colors_2[index2]
                break
            match_colors_2[index2] = colors[color_index]
            match_colors_1[index1] = colors[color_index]
    color_index += 1

print("Number of distinct colors (except gray): ",len(set(match_colors_1))-1)
    

ax.barh(categories, values_left, color=match_colors_1, align='center', label='% CLIP')
ax.barh(categories, [-value for value in values_right], color=match_colors_2, align='center', label='% Unimodal')

# Draw a vertical line at x=0 to separate the two sets of bars
ax.axvline(x=0, color='white', linewidth=5)

# Draw arrows for matching categories
# indent = 1.2
# for y1, y2 in matches:
#     ax.annotate("", xy=(indent, y1), xytext=(-indent, y2),
#                 arrowprops=dict(arrowstyle="<->", color=match_colors_1[y1]))
    
# Set labels, legend, and grid
# ax.set_xlabel('Values')
# ax.set_title('Diverging Bar Chart with Arrows')
# ax.legend()

# Remove the y-ticks
ax.set_yticks([])
ax.set_xticks([])

# Add the category names to the left of the y-axis
max_x_clip = 0
for i, (cat, diff) in enumerate(top_10_differences_clip):
    if diff > max_x_clip:
        max_x_clip = diff
# max_x_clip = top_10_differences_clip[1][1]
max_x_uni = 0
for i, (cat, diff) in enumerate(top_10_differences_uni):
    if diff > max_x_uni:
        max_x_uni = diff
for i, (cat, diff) in enumerate(top_10_differences_clip):
    cat = folder[cat][4:].replace('_', ' ')
    ax.text(0.4, i, f"{cat}", va='center', ha='right', color='black', fontsize=10)
    ax.text(max_x_clip-2, i, f"{diff:.2f}", va='center', ha='right', color='black', fontsize=10)
for i, (cat, diff) in enumerate(top_10_differences_uni):
    cat = folder[cat][4:].replace('_', ' ')
    ax.text(-0.4, i, f"{cat}", va='center', ha='left', color='black', fontsize=10)
    ax.text(-max_x_uni+2, i, f"{diff:.2f}", va='center', ha='left', color='black', fontsize=10)


ax.set_xlim(-max_x_uni, max_x_clip)
# Invert x-axis to have the highest values closest to the y-axis
ax.invert_xaxis()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('bird_alignment_clip_vs_uni.pdf')
# Show the plot
plt.show()

# %%
# draw red circle on images
import os, cv2
import numpy as np
from PIL import Image

test_folder = '/home/tin/datasets/cub/CUB/test/'
mask_folder = '/home/tin/datasets/cub/CUB/segmentations/'
save_folder = '/home/tin/datasets/cub/CUB/test_red_circle/'

label_folders = os.listdir(test_folder)

for f in label_folders:
    if not os.path.exists(f'{save_folder}/{f}'):
        os.makedirs(f'{save_folder}/{f}')

    image_paths = os.listdir(f"{test_folder}/{f}")

    for image_path in image_paths:
        full_image_path = f"{test_folder}/{f}/{image_path}"
        full_mask_path = f"{mask_folder}/{f}/{image_path[:-4]}.png"
        
        image = cv2.imread(full_image_path)
        mask = cv2.imread(full_mask_path, 0)

        # Load images
        # image = np.asarray(Image.open(full_image_path).convert('RGB'))
        # mask = np.asarray(Image.open(full_mask_path).convert('RGB')) / 255
        # mask = np.uint8(255 * mask)  # Scale to 0-255
        # cv2.imwrite('mask_abc.jpg', mask)
        
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # Find the contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        # Assuming we want to fit an ellipse to the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.shape[0] >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(largest_contour)
                # Draw the ellipse on a black canvas
                ellipse_image = np.zeros_like(mask)
                cv2.ellipse(image, ellipse, (0, 0, 255), 2)
            else:
                raise ValueError("Not enough points to fit an ellipse")
    
        
        cv2.imwrite(f'{save_folder}/{f}/{image_path}', image)

        # cv2.imwrite('abc.jpg', image)


# %%
