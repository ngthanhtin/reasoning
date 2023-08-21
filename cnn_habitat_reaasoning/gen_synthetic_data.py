from PIL import Image
import os
import json
import numpy as np
import random
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
import pandas as pd

def get_graph_of_cub():
    cub_path = '/home/tin/datasets/cub/CUB/images/'
    n_clusters = 50
    class_cub_cluster_path = f'../plain_clip/class_cub_clusters_{n_clusters}.json'
    
    f = open(class_cub_cluster_path, 'r')
    cluster_data = json.load(f)

    # graph cluster
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
    return graph


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

def get_graph_of_nabirds():
    """
    return graph = {'0297': ['0295', '0294', etc], etc}
    """

    nabirds_path = '/home/tin/datasets/nabirds/images/'
    n_clusters = 100 #196
    class_nabirds_cluster_path = f'../plain_clip/class_nabirds_clusters_{n_clusters}.json'
    
    f = open(class_nabirds_cluster_path, 'r')
    cluster_data = json.load(f)

    graph = {}
    # labelname2labelidx = {}
    # labelname2labelidx = nabirds_classes.set_index('name')['index'].to_dict()

    for k,v in cluster_data.items():
        for label_name in v:
            label_idx = labelname2labelidx[label_name]
            label_idx = str(label_idx)
            if len(label_idx) == 2:
                label_idx = '00' + label_idx
            elif len(label_idx) == 3:
                label_idx = '0' + label_idx

            if label_idx not in graph:
                graph[label_idx] = []
            for label_name2 in v:
                vertice = labelname2labelidx[label_name2]
                vertice = str(vertice)
                if len(vertice) == 2:
                    vertice = '00' + vertice
                elif len(vertice) == 3:
                    vertice = '0' + vertice
                graph[label_idx].append(vertice)

    return graph

def mask_image(file_path, out_dir_name, remove_bkgnd=True):
    """
    Remove background or foreground using segmentation label
    """
    im = np.array(Image.open(file_path).convert('RGB'))
    segment_path = file_path.replace('images', 'segmentations').replace('.jpg', '.png')
    segment_im = np.array(Image.open(segment_path).convert('L'))
    #segment_im = np.tile(segment_im, (3,1,1)) #3 x W x H
    #segment_im = np.moveaxis(segment_im, 0, -1) #W x H x 3
    mask = segment_im.astype(float)/255
    if not remove_bkgnd: #remove bird in the foreground instead
        mask = 1 - mask
    new_im = (im * mask[:, :, None]).astype(np.uint8)
    Image.fromarray(new_im).save(file_path.replace('/images/', out_dir_name))

def mask_dataset(test_pkl, out_dir_name, remove_bkgnd=True):
    data = pickle.load(open(test_pkl, 'rb'))
    file_paths = [d['img_path'] for d in data]
    for file_path in file_paths:
        mask_image(file_path, out_dir_name, remove_bkgnd)

def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.ANTIALIAS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            # assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.ANTIALIAS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.ANTIALIAS)
    return source_resized


def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask

    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined

def get_random_subset(input_list, subset_size):
    if subset_size > len(input_list):
        raise ValueError("Subset size cannot be greater than the length of the input list.")
    
    random_subset = random.sample(input_list, subset_size)
    return random_subset

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Make segmentations',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='nabirds', help='cub or nabirds')
    parser.add_argument('--places_dir', default='/home/tin/datasets/cub/CUB_inpaint_all_train/', help='Path to Places dataset')
    parser.add_argument('--out_dir', default='/home/tin/datasets/cub/CUB_irrelevant_augmix_train/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # cub
    if args.dataset == 'cub':
        img_dir = '/home/tin/datasets/cub/CUB/train/'#os.path.join(args.cub_dir, 'images')
        seg_dir = '/home/tin/datasets/cub/CUB/segmentations/'#os.path.join(args.cub_dir, 'segmentations')
        args.places_dir = '/home/tin/datasets/cub/CUB_inpaint_all_train/'
        args.out_dir = "/home/tin/datasets/cub/temp_gen_data/CUB_irrelevant_with_orig_birds_train"
        # args.out_dir = '/home/tin/datasets/cub/CUB_irrelevant_augmix_train/'

        graph = get_graph_of_cub()
    # nabirds
    elif args.dataset == 'nabirds':
        img_dir = '/home/tin/datasets/nabirds/train/'
        seg_dir = '/home/tin/datasets/nabirds/gen_data/mask_images/'
        args.places_dir = '/home/tin/datasets/nabirds/gen_data/inpaint_images/train_inpaint/'
        args.out_dir = '/home/tin/datasets/nabirds/gen_data/temp_gen_data/augirrelevant_with_orig_birds_train/'
        # args.out_dir = '/home/tin/datasets/nabirds/gen_data/augmix_images_100/'

        graph = get_graph_of_nabirds()

    args.save_black_bg_img = False
    args.augsame = False
    args.augmix = False
    args.augirrelevant = True

    # Make output directory
    os.makedirs(args.out_dir, exist_ok=True)

    label_folders = os.listdir(img_dir)
    label_folders = sorted(label_folders)
    
    for folder in tqdm(label_folders):
        if args.dataset == 'cub':
            folder_index = int(folder.split('.')[0])
            clusters = graph[folder_index]
        elif args.dataset == 'nabirds':
            clusters = graph[folder]

        # create label folders for out_dir
        if not os.path.exists(f"{args.out_dir}/{folder}"):
            os.makedirs(f"{args.out_dir}/{folder}")
        #
        image_files = os.listdir(f"{img_dir}/{folder}")
        for file in image_files:
            # if os.path.exists(f"{args.out_dir}/{folder}/{file}"):
            #     continue
            full_img_path = f"{img_dir}/{folder}/{file}"
            if args.dataset == 'cub':
                full_seg_path = f"{seg_dir}/{folder}/{file[:-4]}.png"
            if args.dataset == 'nabirds':
                full_seg_path = f"{seg_dir}/{folder}/{file}"

            # Load images
            img_np = np.asarray(Image.open(full_img_path).convert('RGB'))
            # Turn into opacity filter
            seg_np = np.asarray(Image.open(full_seg_path).convert('RGB')) / 255

            # Black background
            img_black_np = np.around(img_np * seg_np).astype(np.uint8)

            full_black_path = f"{args.out_dir}/{folder}/{file}"
            img_black = Image.fromarray(img_black_np)
            if args.save_black_bg_img:
                img_black.save(full_black_path)

            # aug background
            if args.augsame:
                for file2 in tqdm(image_files):
                    # if os.path.exists(f"{args.out_dir}/{folder}/{file[:-4]}_{file2}"):
                    #     continue
                    train_place_path = f"{args.places_dir}/{folder}/{file2}"
                    train_place = Image.open(train_place_path).convert('RGB')

                    img_train = combine_and_mask(train_place, seg_np, img_black)

                    full_train_path = f"{args.out_dir}/{folder}/{file[:-4]}_{file2}"
                    img_train.save(full_train_path)
                    
            if args.augmix or args.augirrelevant:
                
                if args.dataset == 'cub':
                    not_neigbors = [i for i in range(200) if i not in clusters] # for cub
                if args.dataset == 'nabirds':
                    not_neigbors = [i for i in os.listdir(img_dir) if i in clusters] # for nabirds
                # take only 3 irrelevant neighbors:
                try:
                    not_neigbors = get_random_subset(not_neigbors, 3)
                except:
                    not_neigbors = not_neigbors

                # for neigbor in clusters:
                for neigbor in not_neigbors:
                    if args.dataset == 'cub':
                        image_files2 = os.listdir(f"{img_dir}/{label_folders[neigbor-1]}") # if it is cub
                    if args.dataset == 'nabirds':
                        image_files2 = os.listdir(f"{img_dir}/{neigbor}") # if it is nabirds
                    # image_files2 = get_random_subset(image_files2, 2)

                    # get random habitat
                    random_image_file = get_random_subset(image_files2, 1)[0]

                    for file2 in image_files2:
                        file2 = random_image_file
                        if os.path.exists(f"{args.out_dir}/{folder}/{file[:-4]}_{file2}"):
                            continue
                        if args.dataset == 'cub':
                            train_place_path = f"{args.places_dir}/{label_folders[neigbor-1]}/{file2}" # if it is cub
                        if args.dataset == 'nabirds':
                            train_place_path = f"{args.places_dir}/{neigbor}/{file2}" # if it is nabirds
                        # train_place_path = f"{args.places_dir}/{folder}/{file2}"
                        train_place = Image.open(train_place_path).convert('RGB')

                        img_train = combine_and_mask(train_place, seg_np, img_black)

                        # full_train_path = f"{args.out_dir}/{folder}/{file[:-4]}_{file2}"
                        full_train_path = f"{args.out_dir}/{folder}/{file}"
                        img_train.save(full_train_path)
                        break

#%%

