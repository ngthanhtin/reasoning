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
        data[bird] = {'Shape': '', 'Size': '', 'Habitat': ''}
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
# %%
chatgpt_desc_path = "/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json"

# %% -----CHATGPT-----
f = open(chatgpt_desc_path, 'r')
chatgpt_desc_data = json.load(f)

# %% test allaboutbird data and chatgpt data
chatgpt2allaboutbirds_path = '/home/tin/reasoning/plain_clip/overlapped_inat_allaboutbirds.json'
f = open(chatgpt2allaboutbirds_path, 'r')
chatgpt2allaboutbirds_namedict = json.load(f)
# %%

num_overlapped = 0
for chatgpt_name, allaboutbird_name in chatgpt2allaboutbirds_namedict.items():
    chatgpt_desc_data[chatgpt_name] = []
    chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Shape'])
    chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Size'])
    chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Color'])
    chatgpt_desc_data[chatgpt_name].append(allaboutbird_data[allaboutbird_name]['Habitat'])
    num_overlapped += 1

num_overlapped

# %%
chatgpt_desc_data
# %%
# Serializing json
json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json", "w") as outfile:
    outfile.write(json_object)

# %% ----POST PROCESSING----
additional_chatgpt_desc_path = "/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json"

f = open(additional_chatgpt_desc_path, 'r')
chatgpt_desc_data = json.load(f)

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
chatgpt_desc_data = preprocess_descs(chatgpt_desc_data)
# Serializing json
json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json", "w") as outfile:
    outfile.write(json_object)

# %% Remove habitat
desc_path = "/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID_descriptors_inaturalist.json"

f = open(desc_path, 'r')
desc_data = json.load(f)
for bird in desc_data:
    desc_data[bird] = desc_data[bird][:-1]

json_object = json.dumps(desc_data, indent=4)
with open("/home/tin/reasoning/plain_clip/descriptors/inaturalist2021/425_ID2_descriptors_inaturalist.json", "w") as outfile:
    outfile.write(json_object)
# %%
