import torch
import os

import numpy as np
import random
import string

import json
def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)
    

def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}."
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}."
    elif descriptor.startswith('used'):
        return f"which is {descriptor}."
    else:
        return f"which has {descriptor}."
    
# def make_descriptor_sentence(descriptor):
#     return descriptor.replace('It', 'which').replace('.', ',')
    
def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

def generate_adversarial_text(text):
    """ Generate adversarial text by replacing characters with visually similar symbols or altering case. """
    substitutions = {
        'a': '@', 'e': '3', 'i': '!', 'o': '0', 's': '$',
        'A': '4', 'E': '€', 'I': '|', 'O': '()', 'S': '5'
    }
    return ''.join(substitutions.get(c, c) for c in text)

def generate_naturally_corrupted_text(text):
    """ Generate naturally corrupted text by introducing common typos. """
    def replace_random_char(s):
        if s and random.random() < 0.15:  # Roughly 15% chance to modify a character
            random_char = random.choice(string.ascii_lowercase)
            random_index = random.randint(0, len(s) - 1)
            return s[:random_index] + random_char + s[random_index + 1:]
        return s

    return ' '.join(replace_random_char(word) for word in text.split())


def load_gpt_descriptions(hparams, classes_to_load=None, sci_2_comm=None):
    gpt_descriptions_unordered = load_json(hparams['descriptor_fname'])
    unmodify_dict = {}
    
    
    if classes_to_load is not None: 
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered
    if hparams['category_name_inclusion'] is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)
        
        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']
            # v = [generate_adversarial_text(vv) for vv in v]
            # v = [generate_naturally_corrupted_text(vv) for vv in v]

            if sci_2_comm:
                word_to_add = wordify(sci_2_comm[k])
            else:
                word_to_add = wordify(k)
            
            if (hparams['category_name_inclusion'] == 'append'):
                build_descriptor_string = lambda item: f"{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['between_text']}{word_to_add}"
            elif (hparams['category_name_inclusion'] == 'prepend'):
                build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['after_text']}"
                # build_descriptor_string = lambda item: f"Visual Redaction contains {word_to_add} information{hparams['between_text']}{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['after_text']}"

            else:
                build_descriptor_string = lambda item: modify_descriptor(item, hparams['apply_descriptor_modification'])
            
            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
                
            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]
            
            # print an example the first time
            if i == 0: #verbose and 
                print(f"\nExample description for class {k}: \"{gpt_descriptions[k][0]}\"\n")
    
    return gpt_descriptions, unmodify_dict


def seed_everything(seed: int):
    # import random, os
    # import numpy as np
    # import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
 
import matplotlib.pyplot as plt

stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means
  
def show_single_image(image):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    denorm_image = denormalize(image.unsqueeze(0).cpu(), *stats)
    ax.imshow(denorm_image.squeeze().permute(1, 2, 0).clamp(0,1))
    
    plt.show()