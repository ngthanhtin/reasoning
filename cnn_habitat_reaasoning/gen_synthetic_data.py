from PIL import Image
import os
import json
import numpy as np
import random
from collections import defaultdict
import pickle
import random

N_CLASSES = 200

def get_graph():
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
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
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

    parser.add_argument('--cub_dir', default='/home/tin/datasets/cub/CUB/train/', help='Path to CUB (should also contain segmentations folder)')
    parser.add_argument('--places_dir', default='/home/tin/datasets/cub/CUB_inpaint_all_train/', help='Path to Places365 dataset')
    parser.add_argument('--out_dir', default='/home/tin/datasets/cub/CUB_augmix_train/', help='Output directory')
    parser.add_argument('--black_dirname', default='CUB_black', help='Name of black dataset: black background for each image')
    parser.add_argument('--random_dirname', default='CUB_random', help='Name of random dataset: completely random place sampled for each image')
    parser.add_argument('--fixed_dirname', default='CUB_fixed', help='Name of fixed dataset: class <-> place association fixed at train, swapped at test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get species
    img_dir = args.cub_dir#os.path.join(args.cub_dir, 'images')
    seg_dir = '/home/tin/datasets/cub/CUB/segmentations/'#os.path.join(args.cub_dir, 'segmentations')
    species = sorted(os.listdir(img_dir))

    graph = get_graph()

    # Make output directory
    os.makedirs(args.out_dir, exist_ok=True)

    label_folders = os.listdir(img_dir)
    label_folders = sorted(label_folders)
    
    for folder in label_folders:
        folder_index = int(folder.split('.')[0])
        clusters = graph[folder_index]
        #
        if not os.path.exists(f"{args.out_dir}/{folder}"):
            os.makedirs(f"{args.out_dir}/{folder}")
        image_files = os.listdir(f"{img_dir}/{folder}")

    # (image, segmentation, train place, test place
    # it = zip(spc_img, spc_seg, train_place_imgs, test_place_imgs)

    # for img_path, seg_path, train_place_path, test_place_path in it:
        for file in image_files:
            full_img_path = f"{img_dir}/{folder}/{file}"
            full_seg_path = f"{seg_dir}/{folder}/{file[:-4]}.png"

            # Load images
            img_np = np.asarray(Image.open(full_img_path).convert('RGB'))
            # Turn into opacity filter
            seg_np = np.asarray(Image.open(full_seg_path).convert('RGB')) / 255

            # Black background
            img_black_np = np.around(img_np * seg_np).astype(np.uint8)

            # full_black_path = os.path.join(spc_black_dir, img_path)
            img_black = Image.fromarray(img_black_np)
            # img_black.save(full_black_path)

            # Fixed background

            for neigbor in clusters:
                image_files2 = os.listdir(f"{img_dir}/{label_folders[neigbor-1]}")
                image_files2 = get_random_subset(image_files2, 2)

                for file2 in image_files2:
                    train_place_path = f"{args.places_dir}/{label_folders[neigbor-1]}/{file2}"
                    train_place = Image.open(train_place_path).convert('RGB')
                    # test_place = Image.open(test_place_paxth).convert('RGB')

                    img_train = combine_and_mask(train_place, seg_np, img_black)
                    # img_test = combine_and_mask(test_place, seg_np, img_black)

                    full_train_path = f"{args.out_dir}/{folder}/{file[:-4]}_{file2}"
                    img_train.save(full_train_path)
                    # full_test_path = os.path.join(spc_test_dir, img_path)
                    # img_test.save(full_test_path)

#%%

