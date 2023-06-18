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


def get_allaboutbirds_info(path="/home/tin/reasoning/scraping/allaboutbirds_ids/"):
    """
    return: a dict E.g. "Dark-eyed Junco": {
        "Size": "large small",
        "Color": "Dark-eyed Junco",
        "Behavior": "The Dark-eyed Junco is a medium-sized sparrow with a rounded head, a short, stout bill and a fairly long, conspicuous tail.",
        "Habitat": "Example query for example 1",
        },
    """
    birds = os.listdir(path)
    print(f"There are {len(birds)} birds")
    data = {}

    for bird in birds:
        data[bird] = {'Shape': '', 'Size': '', 'Color': '', 'Habitat': ''}
        # read meta file
        meta_json_path = os.path.join(path, f'{bird}/meta.json')

        f = open(meta_json_path)
        bird_data = json.load(f)
        
        # dict_keys(['Shape', 'Compared Size', 'Relative Size', 'Measurements'])
        shape_sentence_list = bird_data['Size']['description']['Shape']
        compared_size_sentence_list = bird_data['Size']['description']['Compared Size']
        relative_size_sentence_list = bird_data['Size']['description']['Relative Size']
        color_sentence_list = bird_data['Color']['description']
        habitat_sentence_list = bird_data['Habitat']['description']
        
        # check list
        # if len(habitat_sentence_list) == 0: # --> There are 11 birds that dont have compared size --> cause the errors
        #     print(bird)

        # preprocessing text
        shape_sentence = preprocess_sentence_list(shape_sentence_list)
        compared_size_sentence = preprocess_sentence_list(compared_size_sentence_list)
        relative_size_sentence = preprocess_sentence_list(relative_size_sentence_list)
        if compared_size_sentence:
            size_sentence = f'{compared_size_sentence} {relative_size_sentence}'
        else:
            size_sentence = f'{relative_size_sentence}'
        
        color_sentence = preprocess_sentence_list(color_sentence_list)
        habitat_sentence = preprocess_sentence_list(habitat_sentence_list)

        data[bird]['Shape'] = f"Shape: {shape_sentence}"
        data[bird]['Size'] = f"Size: {size_sentence}"
        data[bird]['Color'] = f"Color: {color_sentence}"
        data[bird]['Habitat'] = f"Habitat: {habitat_sentence}"

    return data

allaboutbird_path = "/home/tin/reasoning/scraping/allaboutbirds_ids/"
allaboutbird_data = get_allaboutbirds_info(allaboutbird_path)

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
chatgpt_desc_path = "/home/tin/reproduce_visual_descriptors/classify_by_description_release/descriptors/chatgpt_descriptors_nabirds.json"

# %% -----CHATGPT-----
f = open(chatgpt_desc_path, 'r')
chatgpt_desc_data = json.load(f)

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

similarity_score = compute_similarity('Swainson\'s Hawk', 'Swainsons Hawk')
print(similarity_score)
similarity_score = compute_similarity('Common Eider', 'Common Eider')
similarity_score
# %% test allaboutbird data and chatgpt data
chatgpt2allaboutbirds_namedict = {}

num_overlapped = 0
for bird in chatgpt_desc_data:
    fix_name, anno = remove_annotation(bird)
    # if fix_name in allaboutbird_data.keys():

    is_overlapped = False
    for k in allaboutbird_data.keys():
        fix_k = fix_bird_name(k)
        if fix_name == fix_k:
            num_overlapped += 1
            is_overlapped = True
            chatgpt2allaboutbirds_namedict[bird] = k
    
    if not is_overlapped:
        for k in allaboutbird_data.keys():
            fix_k = fix_bird_name(k)
            similarity_score = compute_similarity(fix_name, fix_k)
            if similarity_score > 0.4:
                num_overlapped+=1
                is_overlapped = True
                # print(fix_name, fix_k, similarity_score)
                chatgpt2allaboutbirds_namedict[bird] = k

    if not is_overlapped:
        print(fix_name)
        chatgpt2allaboutbirds_namedict[bird] = ''

# %%
num_overlapped = 0

num_overlapped = 0
for chatgpt_name, allaboutbird_name in chatgpt2allaboutbirds_namedict.items():

    chatgpt_desc_data[chatgpt_name] = []
    if allaboutbird_name != '':
        chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Shape'])
        chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Size'])
        chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Color'])
        chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Habitat'])
        num_overlapped += 1
    else:
        chatgpt_desc_data[chatgpt_name].append("")
        chatgpt_desc_data[chatgpt_name].append("")
        chatgpt_desc_data[chatgpt_name].append("")
        chatgpt_desc_data[chatgpt_name].append("")

print(num_overlapped)

# %%
# Serializing json
json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("ID_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)

# %% ----POST PROCESSING----

desc_path = "/home/tin/xclip/ID_descriptors_nabirds.json"

f = open(desc_path, 'r')
desc_data = json.load(f)

# %%

def make_character_lowercase(string, i):
    # Convert the string to a list
    characters = list(string)

    # Check if the index is within the range of the string
    if 0 <= i < len(characters):
        # Convert the character at index i to lowercase
        characters[i] = characters[i].lower()

    # Convert the list back to a string
    new_string = ''.join(characters)

    return new_string

def preprocess_descs(descs):
    num = 0
    for bird in descs:
        descs[bird][-3] = make_character_lowercase(descs[bird][-3], 0)
        descs[bird][-2] = make_character_lowercase(descs[bird][-2], 0)
        descs[bird][-1] = make_character_lowercase(descs[bird][-1], 0)
        descs[bird][0] = make_character_lowercase(descs[bird][0], 0)
        
    return descs

    


# %%
desc_data = preprocess_descs(desc_data)
# Serializing json
json_object = json.dumps(desc_data, indent=4)
with open("ID_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)

# %%
# What is the relative size of the <class_name> compared to other birds in a short phrase?

missing_bird_f = open('missing_nabirds_birds.json', 'r')
missing_birds = json.load(missing_bird_f)

# %%
# for bird in missing_birds:
#     print(f"Describe the shape of the {bird} in a short phrase")
# #%%
# for bird in missing_birds:
#     print(f"Describe the color of the {bird} in a short phrase")
# %%
import json
desc_path = "/home/tin/xclip/ID_descriptors_nabirds.json"

f = open(desc_path, 'r')
desc_data = json.load(f)


# %%
for mbird in missing_birds:
    desc_data[mbird] = []
    
    shape_desc = "shape: " + missing_birds[mbird]["Shape"][0]
    size_desc = "size: " + missing_birds[mbird]["Size"][0]
    color_desc = "color: " + missing_birds[mbird]["Color"][0]
    habitat_desc = "habitat: " + missing_birds[mbird]["Habitat"][0]

    desc_data[mbird].append(shape_desc)
    desc_data[mbird].append(size_desc)
    desc_data[mbird].append(color_desc)
    desc_data[mbird].append(habitat_desc)

# %%
# Serializing json
json_object = json.dumps(desc_data, indent=4)
with open("ID_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)

# %% - REMOVE HABITAT
import json
desc_path = "/home/tin/xclip/ID_descriptors_nabirds.json"

f = open(desc_path, 'r')
desc_data = json.load(f)

for bird in desc_data:
    desc_data[bird] = desc_data[bird][:-1]

json_object = json.dumps(desc_data, indent=4)
with open("ID2_descriptors_nabirds.json", "w") as outfile:
    outfile.write(json_object)
# %%
