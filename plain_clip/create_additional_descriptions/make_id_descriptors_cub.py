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
    name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    return name
# %%
desc_path = "/home/tin/xclip/data/text/sachit/descriptors_cub.json"
# desc_path = '/home/tin/reproduce_visual_descriptors/classify_by_description_release/descriptors/chatgpt_descriptors_nabirds.json'

# %%
# %% Sachit, chatGPT -> Allaboutbirds
severe_mapping = {"Chuck will Widow": "Chuck wills widow",
           "Frigatebird": "Magnificent Frigatebird",
           "Florida Jay": "Florida Scrub Jay",
           "Nighthawk": "Common Nighthawk",
           "Whip poor Will": "Eastern Whip poor will",
           "Le Conte Sparrow": "LeContes Sparrow",
           "Nelson Sharp tailed Sparrow": "Nelsons Sparrow",
           "Artic Tern": "Arctic Tern"}

allaboutbird_change_names = []
allaboutbird_changed_orig_mapping = {}
for bird in allaboutbird_data:
    changed_name = fix_bird_name(bird)
    allaboutbird_changed_orig_mapping[changed_name] = bird
    allaboutbird_change_names.append(changed_name)

# %% -----CONVERT SACHIT CHANGED NAME TO SACHIT ORIGINAL NAME-----
f = open(desc_path, 'r')
desc_data = json.load(f)

changed_names = []
changed_orig_mapping = {}
for bird in desc_data:
    changed_name = fix_bird_name(bird)
    changed_orig_mapping[changed_name] = bird
    changed_names.append(changed_name)

# %%
num_overlapped = 0
for bird in changed_names:
    if bird in severe_mapping:
        change_of_change_bird = severe_mapping[bird]
        if change_of_change_bird in allaboutbird_change_names:
            desc_data[changed_orig_mapping[bird]] = []
            num_overlapped+=1
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Color'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
    else:
        if bird in allaboutbird_change_names:
            num_overlapped+=1
            desc_data[changed_orig_mapping[bird]] = []
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Shape'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Size'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Color'])
            desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Habitat'])
        else:
            first_space_index = bird.find(' ')
            change_of_change_bird = bird[:first_space_index] + 's' + bird[first_space_index:]

            # check again
            if change_of_change_bird not in allaboutbird_change_names:
                print(bird)
            else:
                num_overlapped += 1
                desc_data[changed_orig_mapping[bird]] = []
                desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
                desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
                desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Color'])
                desc_data[changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
print(num_overlapped)

# %%
desc_data
# %%
# Serializing json
json_object = json.dumps(desc_data, indent=4)
with open("ID_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

# %% ----POST PROCESSING----

desc_path = "/home/tin/xclip/ID_descriptors_cub.json"

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
with open("ID_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

# %% --- ADD THE DESC FOR 20 classes -----
# What is the relative size of the <class_name> compared to other birds in a short phrase?

missing_bird_f = open('missing_birds.json', 'r')
missing_birds = json.load(missing_bird_f)

# %%
# for bird in missing_birds:
#     print(f"Describe the shape of the {bird} in a short phrase")
# #%%
# for bird in missing_birds:
#     print(f"Describe the color of the {bird} in a short phrase")
# %%
import json
desc_path = "/home/tin/xclip/ID_descriptors_cub.json"

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
with open("ID_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)
# %%
