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
        
        habitat_sentence = preprocess_sentence_list(habitat_sentence_list)

        data[bird]['Shape'] = f"Shape: {shape_sentence}"
        data[bird]['Size'] = f"Size: {size_sentence}"
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
chatgpt_desc_path = "/home/tin/xclip/data/text/chatgpt/descriptors_cub.json"
sachit_desc_path = "/home/tin/xclip/data/text/sachit/descriptors_cub.json"

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
f = open(sachit_desc_path, 'r')
sachit_desc_data = json.load(f)

sachit_changed_names = []
sachit_changed_orig_mapping = {}
for bird in sachit_desc_data:
    changed_name = fix_bird_name(bird)
    sachit_changed_orig_mapping[changed_name] = bird
    sachit_changed_names.append(changed_name)

# %%
num_overlapped = 0
for bird in sachit_changed_names:
    if bird in severe_mapping:
        change_of_change_bird = severe_mapping[bird]
        if change_of_change_bird in allaboutbird_change_names:
            num_overlapped+=1
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
    else:
        if bird in allaboutbird_change_names:
            num_overlapped+=1
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Shape'])
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Size'])
            sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Habitat'])
        else:
            first_space_index = bird.find(' ')
            change_of_change_bird = bird[:first_space_index] + 's' + bird[first_space_index:]

            # check again
            if change_of_change_bird not in allaboutbird_change_names:
                print(bird)
            else:
                num_overlapped += 1
                sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
                sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
                sachit_desc_data[sachit_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
print(num_overlapped)

# %%
sachit_desc_data
# %%
# Serializing json
json_object = json.dumps(sachit_desc_data, indent=4)
with open("additional_sachit_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

# %%
# %% -----CONVERT CHANGED CHATGPT NAME TO ORIGINAL NAME-----
f = open(chatgpt_desc_path, 'r')
chatgpt_desc_data = json.load(f)

chatgpt_changed_names = []
chatgpt_changed_orig_mapping = {}
for bird in chatgpt_desc_data:
    changed_name = fix_bird_name(bird)
    chatgpt_changed_orig_mapping[changed_name] = bird
    chatgpt_changed_names.append(changed_name)

# %%

num_overlapped = 0
for bird in chatgpt_changed_names:
    if bird in severe_mapping:
        change_of_change_bird = severe_mapping[bird]
        if change_of_change_bird in allaboutbird_change_names:
            num_overlapped+=1
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
    else:
        if bird in allaboutbird_change_names:
            num_overlapped+=1
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Shape'])
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Size'])
            chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[bird]]['Habitat'])
        else:
            first_space_index = bird.find(' ')
            change_of_change_bird = bird[:first_space_index] + 's' + bird[first_space_index:]

            # check again
            if change_of_change_bird not in allaboutbird_change_names:
                print(bird)
            else:
                num_overlapped += 1
                chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Shape'])
                chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Size'])
                chatgpt_desc_data[chatgpt_changed_orig_mapping[bird]].append(allaboutbird_data[allaboutbird_changed_orig_mapping[change_of_change_bird]]['Habitat'])
print(num_overlapped)

# %%
chatgpt_desc_data
# %%
# Serializing json
json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("additional_chatgpt_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

# %% ----POST PROCESSING----

additional_sachit_desc_path = "/home/tin/xclip/additional_sachit_descriptors_cub.json"
additional_chatgpt_desc_path= "/home/tin/xclip/additional_chatgpt_descriptors_cub.json"

f = open(additional_sachit_desc_path, 'r')
sachit_desc_data = json.load(f)

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

def preprocess_descs(descs, type='sachit'):

    num = 0
    for bird in descs:
        
        if type == 'sachit':
            shape_descs = descs[bird][-3]
            size_descs = descs[bird][-2]
            habitat_descs = descs[bird][-1]
            if 'Shape' in shape_descs:
                # descs[bird][-3] = descs[bird][-3].strip('Shape: ')
                # descs[bird][-2] = descs[bird][-2].strip('Size: ')
                # descs[bird][-1] = descs[bird][-1].strip('Habitat: ')
                descs[bird][-3] = make_character_lowercase(descs[bird][-3], 0)
                descs[bird][-2] = make_character_lowercase(descs[bird][-2], 0)
                descs[bird][-1] = make_character_lowercase(descs[bird][-1], 0)
            else:
                print(bird)
                continue
        elif type == 'chatgpt':
            if len(descs[bird]) < 15:
                # print(bird)
                continue
            descs[bird][12] = make_character_lowercase(descs[bird][12], 0)
            descs[bird][13] = make_character_lowercase(descs[bird][13], 0)
            descs[bird][14] = make_character_lowercase(descs[bird][14], 0)
        # print(descs[bird][-1])
    return descs

    


# %%
sachit_desc_data = preprocess_descs(sachit_desc_data, type='sachit')
# Serializing json
json_object = json.dumps(sachit_desc_data, indent=4)
with open("additional_sachit_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

# %%
chatgpt_desc_data = preprocess_descs(chatgpt_desc_data, type='chatgpt')
# Serializing json
json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("additional_chatgpt_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)


# %% --- ADD THE DESC FOR 20 classes -----
# What is the relative size of the <class_name> compared to other birds in a short phrase?

missing_bird_f = open('missing_birds.json', 'r')
missing_birds = json.load(missing_bird_f)
# %%
# for bird in missing_birds:
#     print(f"Describe the habitat of the {bird} in a short phrase")
# %%
import json
additional_sachit_desc_path = "/home/tin/xclip/additional_sachit_descriptors_cub.json"
additional_chatgpt_desc_path= "/home/tin/xclip/additional_chatgpt_descriptors_cub.json"

f = open(additional_sachit_desc_path, 'r')
sachit_desc_data = json.load(f)

f = open(additional_chatgpt_desc_path, 'r')
chatgpt_desc_data = json.load(f)


# %%
for mbird in missing_birds:
    shape_desc = "shape: " + missing_birds[mbird]["Shape"][0]
    size_desc = "size: " + missing_birds[mbird]["Size"][0]
    habitat_desc = "habitat: " + missing_birds[mbird]["Habitat"][0]

    sachit_desc_data[mbird].append(shape_desc)
    sachit_desc_data[mbird].append(size_desc)
    sachit_desc_data[mbird].append(habitat_desc)

    # chatgpt
    chatgpt_desc_data[mbird].append(shape_desc)
    chatgpt_desc_data[mbird].append(size_desc)
    chatgpt_desc_data[mbird].append(habitat_desc)

# %%
# Serializing json
json_object = json.dumps(sachit_desc_data, indent=4)
with open("additional_sachit_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(chatgpt_desc_data, indent=4)
with open("additional_chatgpt_descriptors_cub.json", "w") as outfile:
    outfile.write(json_object)
# %%
