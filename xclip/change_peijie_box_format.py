# %%
import json

desc_path = "/home/tin/xclip/data/text/chatgpt/additional_chatgpt_descriptors_cub.json"

# %%
# %%
f = open(desc_path)
data = json.load(f)

# %%
import json
data = {}
folder = '/home/tin/reasoning/scraping/allaboutbirds_ids/'
class_names = os.listdir(folder)
for cls in class_names:
    json_path = os.path.join(folder, f'{cls}/meta.json')
    f = open(json_path)
    bird_data = json.load(f)
    shape_descs = bird_data["Size"]["description"]["Shape"]
    color_descs = bird_data["Color"]["description"]
    shape_desc = ""
    color_desc = ""
    for desc in shape_descs:
        shape_desc += desc + ' '
    for desc in color_descs:
        color_desc += desc + ' '
    data[cls] = [shape_desc, color_desc]

#%%
# 12 parts
class_names = list(data.keys())
# parts = data[class_names[0]]
# parts = [p.split(":")[0] for p in parts]

# parts = parts[:-3]
parts = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']

parts, len(parts)

# %% convert list to dict
parts_freq_dict = {p: 0 for p in parts}
parts_freq_dict
# %%
def count_word_occurrences(sentence, word):
    words = sentence.split()
    count = 0

    for w in words:
        if w.lower() == word.lower(): # leg in legs, wing in wings, etc
            count += 1

    return count

for k in data.keys():
    # shape_desc = data[k][12]
    # color_desc = data[k][13]
    shape_desc = data[k][0]
    color_desc = data[k][1]
    # for desc in [shape_desc, color_desc]:
    for desc in [color_desc]:
        for p in parts_freq_dict.keys():
            if p == 'beak':
                parts_freq_dict[p] += count_word_occurrences(desc, 'beak')
                parts_freq_dict[p] += count_word_occurrences(desc, 'bill')
                parts_freq_dict[p] += count_word_occurrences(desc, 'beaks')
                parts_freq_dict[p] += count_word_occurrences(desc, 'bills')
            if p == 'leg':
                parts_freq_dict[p] += count_word_occurrences(desc, 'leg')
                parts_freq_dict[p] += count_word_occurrences(desc, 'foot')
                parts_freq_dict[p] += count_word_occurrences(desc, 'legs')
                parts_freq_dict[p] += count_word_occurrences(desc, 'foots')
            if p == 'eyes':
                parts_freq_dict[p] += count_word_occurrences(desc, 'eyes')
                parts_freq_dict[p] += count_word_occurrences(desc, 'eye')
            else:
                parts_freq_dict[p] += count_word_occurrences(desc, p)
                parts_freq_dict[p] += count_word_occurrences(desc, p+'s')

parts_freq_dict
# %% choose top attributes
sorted_dict = sorted(parts_freq_dict.items(), key=lambda x: x[1], reverse=True)
# Get the top keys
top_keys = [item[0] for item in sorted_dict[:5]]  # Choose top 3 keys

print(top_keys)
# %% --- change peijie NABirds format
import torch
import os

embs_path = "/home/tin/xclip/pred_boxes/nabirds/peijie_data/"
embs_files = os.listdir(embs_path)
embs_files = [os.path.join(embs_path, p) for p in embs_files]
# %%
import json
with open("/home/tin/xclip/data/text/chatgpt/additional_chatgpt_descriptors_nabirds.json") as input_file:
        descriptions = json.load(input_file)
class_list = list(descriptions.keys())
class_list[:10]
# %%

save_path = '/home/tin/xclip/pred_boxes/nabirds/formated_data/'
for file in embs_files:
    formated_data = {}
    file_name = file.split("/")[-1]
    
    embs = torch.load(file)
    image_path = embs['image_path']
    label = int(image_path.split("/")[-2])
    
    formated_data['image_id'] = embs['image_id']
    formated_data['image_path'] = embs['image_path']
    formated_data['class_name'] = embs['class_name']
    formated_data['label'] = label
    formated_data['boxes_info'] = {}

    for cls in class_list:
        formated_data['boxes_info'][cls] = {'scores': [1 for _ in range(12)], 'boxes':[], 'labels': list(range(12))}

        for k, v in embs['boxes_info'].items():
            formated_data['boxes_info'][cls]['boxes'].append(v)
        
        
    
    torch.save(formated_data, f"{save_path}{file_name}")
    
# %% test cub emb
embs_path = "/home/tin/xclip/pred_boxes/cub/owl_vit_owlvit-large-patch14_prompt_5_descriptors_chatgpt/"
embs_path = "/home/tin/xclip/pred_boxes/nabirds/formated_data/"
embs_files = os.listdir(embs_path)
embs_files = [os.path.join(embs_path, p) for p in embs_files]
 # %%
test_file = embs_files[0]

emb = torch.load(test_file)
print(emb['boxes_info'])


# %%
emb['image_path']
# %%
emb["image_path"] = emb["image_path"].replace('lab', 'tin')
emb["image_path"]

# %%---convert xclip_scores_logs.json to class accuracy---
# cub
import json

cub_classes_path = "/home/tin/datasets/cub/dataset/CUB/classes.txt"
path = "/home/tin/xclip/results/cub/chatgpt-ViT-B32/boxes_only/06_06_2023-15:43:15_owl_vit_5parts_viz/xclip_scores_logs.json"
f = open(path, 'r')
data = json.load(f)
keys = list(data.keys())
data.keys()

# %%
for k in keys:
    new_k = k.split('_')
    new_k = new_k[:-2]
    string_k = ''
    for i in range(len(new_k)):
        string_k += new_k[i] + '_'
    string_k = string_k[:-1]
    data[string_k] = data.pop(k)
data.keys()
# %%
f = open(cub_classes_path, 'r')
classes = f.readlines()
print(classes[:10])
for i, cls in enumerate(classes):
    dot_index = cls.index('.')
    classes[i] = cls[dot_index+1:-1]
classes[:10]

num_exists = 0
for cls in classes:
    if cls not in data.keys():
        print(cls)
    else:
        num_exists+=1
num_exists
# %% --read file accuracy
gt_list = [0 for i in range(200)]
pd_list = [0 for i in range(200)]

for k in data.keys():
    gt = data[k]['gt_label']
    gt_list[gt] += 1
    pd = data[k]['clip_topk_preds'][0]
    pd = classes.index(pd)
    pd_list[pd] += 1

pd_list


# %%



# %% --- change peijie's INaturalist 2021 format
import torch
import os
import json

# 1. filter boxes in 425 sub classes of INaturalist
with open("/home/tin/projects/reasoning/xclip/data/text/chatgpt/425_additional_chatgpt_descriptors_inaturalist.json") as input_file:
        descriptions = json.load(input_file)
class_list = list(descriptions.keys())
len(class_list), class_list[:10]

# %%
# 2. get the list of box embedding files
embs_path = "/home/tin/writable/inaturalist/train/owlvit-large-patch14_cub-12-parts/data"
embs_files = os.listdir(embs_path)
embs_files = [os.path.join(embs_path, p) for p in embs_files]
len(embs_files)
# %%
test_file = embs_files[0]
emb = torch.load(test_file)
print(emb)
# %%

save_path = '/home/tin/projects/reasoning/xclip/pred_boxes/inaturalist2021/425_classes_data/'

for file in embs_files:
    embs = torch.load(file)
    if embs['class_name'] not in class_list:
        continue
    formated_data = {}
    file_name = file.split("/")[-1]
    image_path = embs['image_path']
    label = class_list.index(embs['class_name'])

    formated_data['image_id'] = embs['image_id']
    formated_data['image_path'] = embs['image_path']
    formated_data['class_name'] = embs['class_name']
    formated_data['label'] = label
    formated_data['boxes_info'] = {}

    for cls in class_list:
        formated_data['boxes_info'][cls] = {'scores': [1 for _ in range(12)], 'boxes':[], 'labels': list(range(12))}

        for k, v in embs['boxes_info'].items():
            formated_data['boxes_info'][cls]['boxes'].append(v)
        
        
    
    torch.save(formated_data, f"{save_path}{file_name}")

# %% test cub emb
embs_path = "/home/tin/xclip/pred_boxes/cub/owl_vit_owlvit-large-patch14_prompt_5_descriptors_chatgpt/"
# embs_path = "/home/tin/xclip/pred_boxes/nabirds/formated_data/"
embs_path = "/home/tin/projects/reasoning/xclip/pred_boxes/inaturalist2021/425_classes_data/"
embs_files = os.listdir(embs_path)
embs_files = [os.path.join(embs_path, p) for p in embs_files]
 # %%
test_file = embs_files[0]
emb = torch.load(test_file)
print(emb['boxes_info'])
print(emb['label'])


# %%
emb['image_path']
# %%
emb["image_path"] = emb["image_path"].replace('lab', 'tin')
emb["image_path"]