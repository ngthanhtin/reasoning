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
    
def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

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

def load_gpt_descriptions_2(opt, classes_to_load=None, sci_2_comm=None, mode: str='clip'):    
    ### Prepare extracted descriptions.
    gpt_descriptions = load_json(opt.descriptor_fname)
    
    # convert sci 2 common names
    if sci_2_comm:
        gpt_descriptions = {wordify(sci_2_comm[k]):v for k,v in gpt_descriptions.items()}
    
    # Replace empty descriptor lists if necessary.
    gpt_descriptions = {key: item if len(item) else [''] for key, item in gpt_descriptions.items()}
    
    ### (Lazy - uses gpt descriptions) Use the default CLIP setup.
    if not 'label_to_classname' in vars(opt):
        opt.label_to_classname = list(gpt_descriptions.keys())
        opt.n_classes = len(opt.label_to_classname)
        
    ### (Lazy - uses gpt descriptions) Use the default CLIP setup.    
    if mode == 'clip':
        gpt_descriptions = {l: opt.label_before_text + wordify(l) + opt.label_after_text for l in opt.label_to_classname}
    
    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt_descriptions.values()]
    descr_list = np.array([x for y in descr_list for x in y])
    # List of available classes.
    key_list = list(gpt_descriptions.keys())                                       
    
    ### Descriptor Makers.
    structured_descriptor_builder = lambda item, cls: f"{opt.pre_descriptor_text}{opt.label_before_text}{wordify(cls)}{opt.descriptor_separator}{modify_descriptor(item, opt.apply_descriptor_modification)}{opt.label_after_text}"    
    # generic_descriptor_builder = lambda item, cls: f"{opt.pre_descriptor_text}{opt.label_before_text}{wordify(cls)}{opt.descriptor_separator}{item}{opt.label_after_text}"    
    
    # classname + habitat descriptions
    if mode == 'clip_habitat':
        gpt_descriptions = {key: [structured_descriptor_builder(item, key) for item in class_descr_list] for key, class_descr_list in gpt_descriptions.items()}
        gpt_descriptions = {key: [class_descr_list[-1]] for key, class_descr_list in gpt_descriptions.items()}
        gpt_descriptions = {key: [opt.label_before_text + wordify(key) + opt.label_after_text, class_descr_list[-1]] for key, class_descr_list in gpt_descriptions.items()}
        
        if 'habitat' not in opt.descriptor_fname:
            print("The descriptions do not contain habitat descriptions. Exit...")
            exit()

    ### Use description-based CLIP (DCLIP).
    if mode == 'gpt_descriptions':
        gpt_descriptions = {key: [structured_descriptor_builder(item, key) for item in class_descr_list] for key, class_descr_list in gpt_descriptions.items()}

    if 'waffle' in mode:
        import pickle as pkl
        seed_everything(opt.seed)
        word_list = pkl.load(open('/home/tin/projects/reasoning/plain_clip/word_list.pkl', 'rb'))

        descr_list = [list(values)[-1] for values in gpt_descriptions.values()] # get only one habitat description
        descr_list = np.array([x for y in descr_list for x in y])

        avg_num_words = int(np.max([np.round(np.mean([len(wordify(x).split(' ')) for x in key_list])), 1]))
        avg_word_length = int(np.round(np.mean([np.mean([len(y) for y in wordify(x).split(' ')]) for x in key_list])))        
        word_list = [x[:avg_word_length] for x in word_list]

        # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
        character_list = [x.split(' ') for x in descr_list]
        character_list = [x.replace(',', '').replace('.', '') for x in np.unique([x for y in character_list for x in y])]
        character_list = np.unique(list(''.join(character_list)))
        
        #TODO: # replace x by the habitat description which is gpt_descriptions[x][-1]
        num_spaces = int(np.round(np.mean([np.sum(np.array(list(x)) == ' ') for x in key_list]))) + 1 
        num_chars = int(np.ceil(np.mean([np.max([len(y) for y in x.split(' ')]) for x in key_list])))
            
        num_chars += num_spaces - num_chars%num_spaces
        sample_key = ''
        
        for s in range(num_spaces):
            for _ in range(num_chars//num_spaces):
                sample_key += 'a'
            if s < num_spaces - 1:
                sample_key += ' '

        original_gpt_descriptions = gpt_descriptions.copy()        
        gpt_descriptions = {key: [] for key in gpt_descriptions.keys()}
        
        for key in key_list:
            for _ in range(opt.waffle_count):
                # random words
                base_word = ''            
                avg_num_words = len(wordify(original_gpt_descriptions[key][-1]).split(' '))
                for a in range(avg_num_words):
                    base_word += np.random.choice(word_list, 1, replace=False)[0]
                    if a < avg_num_words - 1:
                        base_word += ' '
                gpt_descriptions[key].append(structured_descriptor_builder(base_word, key))
                
                #random characters
                # noise_word = ''                
                # use_key = sample_key if len(key) >= len(sample_key) else key
                # for c in sample_key:
                #     if c != ' ':
                #         noise_word += np.random.choice(character_list, 1, replace=False)[0]
                #     else:
                #         noise_word += ', '
                # gpt_descriptions[key].append(structured_descriptor_builder(noise_word, key))
                                
        match_key = np.random.choice(key_list)
        # gpt_descriptions = {key: gpt_descriptions[match_key] for key in key_list}
        gpt_descriptions = {key: gpt_descriptions[key] for key in key_list}
        
        for key in gpt_descriptions:
            if mode == 'waffle':
                gpt_descriptions[key] = [x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]]
            elif mode == 'waffle_habitat':
                gpt_descriptions[key] = [x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]]
                gpt_descriptions[key].append(structured_descriptor_builder(original_gpt_descriptions[key][-1], key))
            elif mode == 'waffle_habitat_only':
                gpt_descriptions[key] = [structured_descriptor_builder(original_gpt_descriptions[key][-1], key)]
    
    return gpt_descriptions, []


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