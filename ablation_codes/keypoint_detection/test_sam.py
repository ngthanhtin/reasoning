# %%
import argparse
import os
import copy
import json
import numpy as np
import torch
import time
from PIL import Image, ImageDraw, ImageFont

# segment anything
from segment_anything import sam_model_registry, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

def area(box):
    _, _, w, h = box
    return w*h

def is_box_inside(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1+w1, y1_1+h1
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2+w2, y1_2+h2

    if x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2:
        return True
    else:
        return False
    
def calculate_iou(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Calculate the coordinates of the intersection rectangle.
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Check for no intersection (negative area of intersection).
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    # Calculate the areas of the two boxes and the intersection.
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate IoU.
    iou = area_intersection / (area_box1 + area_box2 - area_intersection)
    return iou

def find_intersection_point_box(point, box):
    # make sure the point is inside the box
    x, y = point
    x1, y1, x2, y2 = box

    # Calculate the coordinates of the intersection point for the vertical line.
    if x-x1 < x2-x:
        vertical_intersection_x = x1
    elif x-x1 == x2 - x:
        vertical_intersection_x = x1
    else:
        vertical_intersection_x = x2

    # Calculate the coordinates of the intersection point for the horizontal line.
    if y-y1 < y2 - y:
        horizontal_intersection_y = y1
    elif  -y1> y2-y:
        horizontal_intersection_y = y1
    else:
        horizontal_intersection_y = y2

    return (vertical_intersection_x, horizontal_intersection_y)

def get_mask_object_bbox(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    y, y_end = np.where(rows)[0][[0, -1]]
    x, x_end = np.where(cols)[0][[0, -1]]
    width = x_end - x + 1
    height = y_end - y + 1
    return x, y, width, height

def find_largest_connected_component(mask):
    # Find connected components in the mask
    mask = mask.astype(np.uint8) * 255
    # ret, thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Find the largest connected component (excluding the background label)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # +1 because we skip the background label

    # Create a mask containing only the largest connected component
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255

    return largest_component_mask

# Define a function to check if a point is within the mask
def is_inside_mask(point, binary_mask):
    x, y = point
    return binary_mask[y, x] == 255

# Define a function to find the intersection point of a horizontal line
def find_horizontal_intersection(start_point, binary_mask):
    x, y = start_point
    for x_candidate in range(x, binary_mask.shape[1]):
        if is_inside_mask((x_candidate, y), binary_mask):
            return x_candidate, y
    return None

# Define a function to find the intersection point of a vertical line
def find_vertical_intersection(start_point, binary_mask):
    x, y = start_point
    for y_candidate in range(y, binary_mask.shape[0]):
        if is_inside_mask((x, y_candidate), binary_mask):
            return x, y_candidate
    return None

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

# %%
# cfg
# image_path = '/home/tin/datasets/cub/CUB//images/079.Belted_Kingfisher/Belted_Kingfisher_0006_70625.jpg'
# relative_image_path = '079.Belted_Kingfisher/Belted_Kingfisher_0006_70625.jpg'

# image_path = '/home/tin/datasets/cub/CUB//images/088.Western_Meadowlark/Western_Meadowlark_0063_77946.jpg'
# relative_image_path = '088.Western_Meadowlark/Western_Meadowlark_0063_77946.jpg'

# image_path = '/home/tin/datasets/cub/CUB/images/100.Brown_Pelican/Brown_Pelican_0095_94290.jpg'
# relative_image_path = '100.Brown_Pelican/Brown_Pelican_0095_94290.jpg'

image_path = '/home/tin/datasets/cub/CUB//images/124.Le_Conte_Sparrow/Le_Conte_Sparrow_0102_795195.jpg' 
relative_image_path = '124.Le_Conte_Sparrow/Le_Conte_Sparrow_0102_795195.jpg'

# %%
# load mask
mask_folder = '/home/tin/datasets/cub/CUB/segmentations'

def get_binary_mask(mask_image_path):
    mask_image = cv2.imread(mask_image_path)
    # Load images
    img_np = np.asarray(Image.open(image_path).convert('RGB'))
    # Turn into opacity filter
    seg_np = np.asarray(Image.open(mask_image_path).convert('RGB')) / 255
    # Black background
    mask_image = np.around(img_np * seg_np).astype(np.uint8)
    gray_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # Threshold the mask image to create a binary mask (0 for non-mask, 255 for mask)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    return binary_mask

# %%
# load keypoint dicts
# CUB_ROOT = "/home/tin/datasets/cub/CUB/images/"
f = open("/home/tin/projects/reasoning/keypoint_detection/unsup-parts/unsup-parts/evaluation/cub_eval_kp.json")
kp_data = json.load(f)
# part names
part_label_dict = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'left eye', 'left leg', 'left wing', 'nape', 'right eye', 'right leg', 'right wing', 'tail', 'throat']
color_dict = {
    'b': (255, 0, 0),         # Blue, back
    'g': (0, 255, 0),         # Green, beak
    'r': (0, 0, 255),         # Red, belly
    'c': (255, 255, 0),       # Cyan, breast
    'm': (255, 0, 255),       # Magenta, crown
    'y': (0, 255, 255),       # Yellow, forehead
    'w': (255, 255, 255),     # White, left eye
    'tab:orange': (0, 165, 255),  # Orange, left leg
    'tab:purple': (128, 0, 128),  # Purple, left wing
    'tab:brown': (0, 139, 139),   # Brown, nape
    'tab:pink': (203, 192, 255),  # Pink, right eye
    'tab:gray': (169, 169, 169),  # Gray, right leg
    'lime': (0, 255, 0),         # Lime, right wing
    'aqua': (255, 255, 0),       # Aqua, tail
    'fuchsia': (255, 0, 255)     # Fuchsia, throat
}
# Create a dictionary to map parts to colors
part_to_color = {}
for i, part in enumerate(part_label_dict):
    color_key = list(color_dict.keys())[i % len(color_dict)]
    part_to_color[part] = color_dict[color_key]

# List of BGR color values
bgr_colors = list(color_dict.values())

# %%
# run segment anything (SAM)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:6"
t1 = time.time()
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Time to load SAM: ", time.time()-t1)
# %%
image = cv2.imread(image_path)
h,w = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# transform image
pil_img = Image.open(image_path)
transform = T.Resize(size = (256))
pil_image = transform(pil_img)
new_w, new_h = pil_image.size

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
# %%
orig_image = cv2.imread(image_path)
predictor.set_image(image)

for kp_key in kp_data:
    relative_kp_key = kp_key.split("/")[-2] + "/" + kp_key.split("/")[-1]
    if relative_image_path == relative_kp_key:
        mask_img_path = f"{mask_folder}/{relative_image_path[:-4]}.png"
        binary_mask = get_binary_mask(mask_img_path)
        bird_box = get_mask_object_bbox(binary_mask)
        
        parts = kp_data[kp_key]

        list_coords = []
        for i, part_name in enumerate(part_label_dict):
            x,y = parts[part_name]
            x,y = int(x), int(y)
            x,y = (x/new_w)*w, (y/new_h)*h
            list_coords.append([x,y])
        # ----
        for k, (part_name, xy) in enumerate(parts.items()):
            x,y = int(xy[0]), int(xy[1])
            if x < 0 or y < 0:
                continue
            x,y = (x/new_w)*w, (y/new_h)*h

            labels = [0 for i in range(len(part_label_dict))]
            labels[part_label_dict.index(part_name)] = 1
            # if part_name == 'belly':
            #     labels[part_label_dict.index('left leg')] = 1
            #     labels[part_label_dict.index('right leg')] = 1
            if part_name == 'beak':
                labels[part_label_dict.index('crown')] = 1
                labels[part_label_dict.index('forehead')] = 1
            if part_name == 'crown':
                labels[part_label_dict.index('left eye')] = 1
                labels[part_label_dict.index('right eye')] = 1
            if part_name == 'forehead':
                labels[part_label_dict.index('crown')] = 1
            if part_name == 'left leg':
                labels[part_label_dict.index('right leg')] = 1
            if part_name == 'right leg':
                labels[part_label_dict.index('left leg')] = 1
            if part_name == 'left eye':
                labels[part_label_dict.index('right eye')] = 1
            if part_name == 'right eye':
                labels[part_label_dict.index('left eye')] = 1
            if part_name == 'left wing':
                labels[part_label_dict.index('right wing')] = 1
            if part_name == 'right wing':
                labels[part_label_dict.index('left wing')] = 1
            
            if part_name in ['beak', 'tail', 'right leg', 'left leg', 'right eye', 'left eye']: # beak only
                x,y = int(x), int(y)
                if not is_inside_mask((x,y), binary_mask):
                    horizontal_intersection = find_horizontal_intersection((x,y), binary_mask)
                    vertical_intersection = find_vertical_intersection((x,y), binary_mask)
                    if horizontal_intersection and vertical_intersection:
                        if abs(x-horizontal_intersection[0]) < abs(y-vertical_intersection[1]):
                            x = horizontal_intersection[0]
                        else:
                            y = vertical_intersection[1]
                    else:
                        if horizontal_intersection:
                            x = horizontal_intersection[0]
                        else:
                            y = vertical_intersection[1]

                input_point = np.array([[x,y]])
                input_label = np.array([1])   
                intersect_x, intersect_y = find_intersection_point_box((x,y), [bird_box[0], bird_box[1], bird_box[0] + bird_box[2], bird_box[1] + bird_box[3]])
                # input_box = np.array([x-10, y-10, x+10, y+10])
                if part_name == 'tail':
                    masks_by_point, scores, logits = predictor.predict(
                        point_coords = input_point,
                        point_labels = input_label,
                        multimask_output = False,
                    )
                    masks = find_largest_connected_component(masks_by_point[0])
                    box_by_point = get_mask_object_bbox(masks)
                    
                    input_box = np.array([min(x, intersect_x), y, max(x, intersect_x), y+ abs(x-intersect_x)])
                    masks_by_box, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    masks = find_largest_connected_component(masks_by_box[0])
                    box_by_box = get_mask_object_bbox(masks)
                    if area(box_by_point) > area(box_by_box):
                        masks = masks_by_point
                    else:
                        masks = masks_by_box
                    
                    # fiter box
                    if calculate_iou(box_by_point, bird_box) > 0: # overlapped
                        if area(box_by_point)/area(bird_box) > 0.7:
                            masks = masks_by_box
                elif part_name in ['right leg', 'left leg']:
                    input_box = np.array([x, min(y, intersect_y), x + abs(y-intersect_y), max(y, intersect_y)])
                    masks, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                elif part_name in ['beak']:
                    masks_by_point, scores, logits = predictor.predict(
                        point_coords = input_point,
                        point_labels = input_label,
                        multimask_output = False,
                    )
                    masks = find_largest_connected_component(masks_by_point[0])
                    box_by_point = get_mask_object_bbox(masks)

                    input_box = np.array([min(x, intersect_x), y, max(x, intersect_x), y+ abs(x-intersect_x)])
                    masks_by_box, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    masks = find_largest_connected_component(masks_by_box[0])
                    box_by_box = get_mask_object_bbox(masks)
                    if calculate_iou(box_by_point, bird_box) > 0:
                        masks = masks_by_point
                    else:
                        masks = masks_by_box
                    if is_box_inside(box_by_point, bird_box):
                        masks = masks_by_point
                elif part_name in ['right eye', 'left eye']:
                    masks_by_point, scores, logits = predictor.predict(
                        point_coords = input_point,
                        point_labels = input_label,
                        multimask_output = False,
                    )
                    masks = find_largest_connected_component(masks_by_point[0])
                    box_by_point = get_mask_object_bbox(masks)

                    input_box = np.array([min(x, intersect_x), y, max(x, intersect_x), y+ abs(x-intersect_x)])
                    masks_by_box, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    masks = find_largest_connected_component(masks_by_box[0])
                    box_by_box = get_mask_object_bbox(masks)
                    if calculate_iou(box_by_point, bird_box) > 0:
                        masks = masks_by_point
                    else:
                        masks = masks_by_box
                    
                    masks = masks_by_box
            else:
                # Find the index of the sublist containing [-1, -1]
                indices_to_remove = []
                copied_list_coords = list_coords.copy()
                copied_labels = labels.copy()
                for i, sublist in enumerate(copied_list_coords):
                    a, b = sublist[0], sublist[1]
                    if a < 0 or b < 0:
                        indices_to_remove.append(i)
                
                for i in reversed(indices_to_remove):
                    del copied_list_coords[i]
                    del copied_labels[i]

                new_copied_list_coords = []
                for p in copied_list_coords:
                    x,y = p
                    p = int(p[0]), int(p[1])
                    if not is_inside_mask(p, binary_mask):
                        horizontal_intersection = find_horizontal_intersection(p, binary_mask)
                        vertical_intersection = find_vertical_intersection(p, binary_mask)
                        if horizontal_intersection and vertical_intersection:
                            if abs(x-horizontal_intersection[0]) < abs(y-vertical_intersection[1]):
                                x = horizontal_intersection[0]
                            else:
                                y = vertical_intersection[1]
                        else:
                            if horizontal_intersection:
                                x = horizontal_intersection[0]
                            else:
                                y = vertical_intersection[1]
                    p = [x,y]
                    new_copied_list_coords.append(p)

                # new_copied_list_coords.append((10, 10))
                # new_copied_list_coords.append((w-10, h-10))
                # copied_labels.append(0)
                # copied_labels.append(0)
                input_point = np.array(new_copied_list_coords)
                input_label = np.array(copied_labels)

                masks, scores, logits = predictor.predict(
                    point_coords = input_point,
                    point_labels = input_label,
                    multimask_output = False,
                )
            masks = find_largest_connected_component(masks[0])
            box = get_mask_object_bbox(masks)
            # fiter box
            if calculate_iou(box, bird_box) > 0: # overlapped
                if area(box)/area(bird_box) > 0.7:
                    continue

            plt.figure(figsize=(10,10))
            cv2.putText(orig_image, part_name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, part_to_color[part_name], 1)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), part_to_color[part_name], 1)
            
        # show_mask(masks, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()  
        plt.clf()
            
        break

# %%
