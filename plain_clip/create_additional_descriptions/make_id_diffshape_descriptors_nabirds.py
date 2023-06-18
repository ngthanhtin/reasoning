# %%
import os
import json

def preprocess_sentence_list(sentence_list):
    sentence_list = [" ".join(s.split()) for s in sentence_list]
    sentence = " ".join(sentence_list)
    if sentence:
        if sentence[-1] == ' ':
            sentence = sentence[:-1]
        if sentence[-1] == '.':
            sentence = sentence[:-1]

    return sentence

# %%
def get_allaboutbirds_meta_info(path="/home/tin/reasoning/scraping/allaboutbirds_ids/"):
    """
    return: a dict E.g. "Dark-eyed Junco": {
        "Size": "large small",
        "Color": "Dark-eyed Junco",
        "Behavior": "The Dark-eyed Junco is a medium-sized sparrow with a rounded head, a short, stout bill and a fairly long, conspicuous tail.",
        "Habitat": "Example query for example 1",
        },
    """
    birds = os.listdir(path)
    data = {}

    for bird in birds:
        data[bird] = {'Shape': '', 'Size': '', 'Habitat': ''}
        # read meta file
        meta_json_path = os.path.join(path, f'{bird}/meta.json')

        f = open(meta_json_path)
        bird_data = json.load(f)
        
        # dict_keys(['Shape', 'Compared Size', 'Relative Size', 'Measurements'])
        shape_sentence_list = bird_data['Size']['description']['Shape']
        compared_size_sentence_list = bird_data['Size']['description']['Compared Size']
        relative_size_sentence_list = bird_data['Size']['description']['Relative Size']
        habitat_sentence_list = bird_data['Habitat']['description']

        # preprocessing text
        shape_sentence = preprocess_sentence_list(shape_sentence_list)
        compared_size_sentence = preprocess_sentence_list(compared_size_sentence_list)
        relative_size_sentence = preprocess_sentence_list(relative_size_sentence_list)
        if compared_size_sentence:
            size_sentence = f'{compared_size_sentence} {relative_size_sentence}'
        else:
            size_sentence = f'{relative_size_sentence}'
        
        habitat_sentence = preprocess_sentence_list(habitat_sentence_list)

        data[bird]['Shape'] = f"shape: {shape_sentence}"
        data[bird]['Size'] = f"size: {size_sentence}"
        data[bird]['Habitat'] = f"habitat: {habitat_sentence}"

    return data

# %%
def get_allaboutbirds_birdtype_info(path="/home/tin/reasoning/scraping/allaboutbirds_ids/"):
    """
    return: a dict E.g. "Dark-eyed Junco": {
        "Size": "large small",
        "Color": "Dark-eyed Junco",
        "Behavior": "The Dark-eyed Junco is a medium-sized sparrow with a rounded head, a short, stout bill and a fairly long, conspicuous tail.",
        "Habitat": "Example query for example 1",
        },
    """
    birds = os.listdir(path)
    data = {}

    for bird in birds:
        data[bird] = {}
        # read bird type file
        birdtype_json_path = os.path.join(path, f'{bird}/bird_type_dict.json')

        f = open(birdtype_json_path)
        bird_data = json.load(f)

        for anno in bird_data:
            data[bird][anno] = ''
            shape_sentence_list = [sent for sent in bird_data[anno]['description'] if sent != '']

            # preprocessing text
            shape_sentence = preprocess_sentence_list(shape_sentence_list)

            data[bird][anno] = f"shape: {shape_sentence}" if shape_sentence != '' else ''

    return data

allaboutbird_path = "/home/tin/reasoning/scraping/allaboutbirds_ids/"
allaboutbird_birdtype_data = get_allaboutbirds_birdtype_info(allaboutbird_path)
allaboutbird_meta_data = get_allaboutbirds_meta_info(allaboutbird_path)

allaboutbird_birdtype_data['Ruddy_Duck']

# %% If there is no description for a specific annotation, I will assign a general description for that type of annotation
for bird in allaboutbird_birdtype_data.keys():
    for ann in allaboutbird_birdtype_data[bird]:
        if allaboutbird_birdtype_data[bird][ann] == '':
            allaboutbird_birdtype_data[bird][ann] = allaboutbird_meta_data[bird]['Shape']

allaboutbird_birdtype_data['Ruddy_Duck']['Breeding male']
# %% Fix bird name
def fix_bird_name(name):
    # name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    return name

def remove_annotation(name):
    if '(' in name:
        cut_idx = name.find('(')
        return name[:cut_idx-1], name[cut_idx:]
    return name, ''
# %%
id_desc_path = "/home/tin/reproduce_visual_descriptors/classify_by_description_release/descriptors/ID_descriptors_nabirds.json"

# %% -----ID 1-----
f = open(id_desc_path, 'r')
id_desc_data = json.load(f)

# %%
import Levenshtein

def compute_similarity(sentence1, sentence2):
    # Tokenization
    tokens1 = sentence1.lower().split()
    tokens2 = sentence2.lower().split()

    # Calculate Levenshtein distance
    distance = Levenshtein.distance(" ".join(tokens1), " ".join(tokens2))

    # Compute similarity score
    similarity = 1 - (distance / max(len(tokens1), len(tokens2)))

    return similarity

similarity_score = compute_similarity('Female/juvenile', 'Female and juvenile')
print(similarity_score)
similarity_score = compute_similarity('Adult dark morph', 'light morph')
similarity_score
# %% test allaboutbird data and chatgpt data
id2allaboutbirds_namedict = {}

num_overlapped = 0
for bird in id_desc_data:
    fix_name, anno = remove_annotation(bird)
    # if fix_name in allaboutbird_data.keys():

    is_overlapped = False
    for k in allaboutbird_birdtype_data.keys():
        fix_k = fix_bird_name(k)
        if fix_name == fix_k:
            num_overlapped += 1
            is_overlapped = True
            id2allaboutbirds_namedict[bird] = k
    
    if not is_overlapped:
        for k in allaboutbird_birdtype_data.keys():
            fix_k = fix_bird_name(k)
            similarity_score = compute_similarity(fix_name, fix_k)
            if similarity_score > 0.4:
                num_overlapped+=1
                is_overlapped = True
                # print(fix_name, fix_k, similarity_score)
                id2allaboutbirds_namedict[bird] = k

    if not is_overlapped:
        print(fix_name)
        id2allaboutbirds_namedict[bird] = ''
        
# %%

num_overlapped = 0
num_changes = 0
changed_birds = []
for chatgpt_name, allaboutbird_name in id2allaboutbirds_namedict.items():

    if allaboutbird_name != '':
        num_overlapped += 1
        for ann in allaboutbird_birdtype_data[allaboutbird_name]:
            # if ann in chatgpt_name:
            if '(' not in chatgpt_name:
                ann_in_chatgpt_name = chatgpt_name
            else:
                first_bracket_idx = chatgpt_name.index('(')
                second_bracket_idx = chatgpt_name.index(')')
                ann_in_chatgpt_name = chatgpt_name[first_bracket_idx+1:second_bracket_idx]
            if compute_similarity(ann, ann_in_chatgpt_name) <= 1 and compute_similarity(ann, ann_in_chatgpt_name) >= -1:
                print(id_desc_data[chatgpt_name][0])
                id_desc_data[chatgpt_name][0] = allaboutbird_birdtype_data[allaboutbird_name][ann]
                print(id_desc_data[chatgpt_name][0])
                print('------')
                num_changes += 1
                changed_birds.append(chatgpt_name)
print("Changes: ", num_changes)
print(num_overlapped)

# %%
changed_birds
# %%
id_desc_data
# %%
# Serializing json
json_object = json.dumps(id_desc_data, indent=4)
with open("ID_diffshape_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)



# %%
# %% - REMOVE HABITAT
import json
desc_path = "ID_diffshape_descriptors_nabirds.json"

f = open(desc_path, 'r')
desc_data = json.load(f)

for bird in desc_data:
    desc_data[bird] = desc_data[bird][:-1]

json_object = json.dumps(desc_data, indent=4)
with open("ID2_diffshape_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)
# %%
