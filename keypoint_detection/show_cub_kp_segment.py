# %%
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

mask_folder = '/home/tin/datasets/cub/CUB/supervisedlabels'

bird_classes = os.listdir(mask_folder)

cls = bird_classes[0]
images = os.listdir(os.path.join(mask_folder, cls))
image_paths = [os.path.join(mask_folder, cls, image) for image in images]

# %%
img = np.asarray(Image.open(image_paths[1]))
imgplot = plt.imshow(img)
plt.show()
# %%
# read OWL-ViT Boxes of CUB
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import cv2

desc_type = 'chatgpt'
boxes_dir = f"/home/tin/projects/reasoning/xclip/pred_boxes/cub/owl_vit_owlvit-large-patch14_prompt_5_descriptors_{desc_type}/"
boxes_files = os.listdir(boxes_dir)
chatgpt_parts = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']
"""
{"image_id": image_ids[i],
"image_path": image_paths[i],
"class_name": gt_class_name,
"label": gt_label,
"boxes_info": boxes_pred_dict}

boxes_pred_dict: {'scores': scores,
                    'boxes': boxes,
                    'labels': list(range(n_descriptors))}
"""
# %% 
# READ BOX
import json

CUB_ROOT = "/home/tin/datasets/cub/CUB/images/"

# load keypoint dicts
f = open("/home/tin/projects/reasoning/keypoint_detection/unsup-parts/unsup-parts/evaluation/cub_eval_kp.json")
kp_data = json.load(f)
part_label_dict = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'left eye', 'left leg', 'left wing', 'nape', 'right eye', 'right leg', 'right wing', 'tail', 'throat']
colors_dict = ['b', 'g', 'r', 'c', 'm',
                   'y', 'w', 'tab:orange', 'tab:purple', 'tab:brown',
                   'tab:pink', 'tab:gray', 'lime', 'aqua', 'fuchsia']
def check_point_in_box(x,y,box):
    """
    box: x1, y1, x2, y2
    """
    x_valid = False
    y_valid = False
    if x >= box[0] and x <= box[2]:
        x_valid = True
    if y >= box[1] and y <= box[3]:
        y_valid = True
    
    if x_valid and y_valid:
        return True
    return False

for file in boxes_files[:10]:
    path = os.path.join(boxes_dir, file)
    owlvit_result = torch.load(path)
    owlvit_result["image_path"] = owlvit_result["image_path"].split("/")[-2] + "/" + owlvit_result["image_path"].split("/")[-1]
    image_path = owlvit_result['image_path']
    print(image_path)
    
    for kp_key in kp_data:
        relative_kp_key = kp_key.split("/")[-2] + "/" + kp_key.split("/")[-1]
        if image_path == relative_kp_key: # filter boxes, draw image
            pil_image = Image.open(kp_key.replace('/dataset/', '/'))
            w,h = pil_image.size

            class_name = owlvit_result['class_name']
            part_from_gpt = owlvit_result['boxes_info'][class_name]['labels']
            boxes = owlvit_result['boxes_info'][class_name]['boxes']
            
            valid_boxes = []
            valid_parts = []

            # transform image
            pil_img = Image.open(kp_key.replace('/dataset/', '/'))
            transform = T.Resize(size = (256))
            pil_image = transform(pil_img)
            new_w, new_h = pil_image.size
            img = np.asarray(Image.open(kp_key.replace('/dataset/', '/')))
            color = (255, 0, 0)
            thickness = 2

            for gpt_part_idx in part_from_gpt:
                part_name = chatgpt_parts[gpt_part_idx]
                if part_name in ['eyes', 'legs', 'wings']:
                    box = boxes[gpt_part_idx]

                    # find part name in kp_data
                    right_kp_x, right_kp_y = kp_data[kp_key][f"right {part_name[:-1]}"]
                    left_kp_x, left_kp_y = kp_data[kp_key][f"left {part_name[:-1]}"]

                    if right_kp_x != -1 and right_kp_y != -1:
                        if check_point_in_box(right_kp_x*w/new_w, right_kp_y*h/new_h, box):
                            valid_boxes.append(box)
                            valid_parts.append(part_name)
                    if left_kp_x != -1 and left_kp_y != -1:
                        if check_point_in_box(left_kp_x*w/new_w, left_kp_y*h/new_h, box):
                            valid_boxes.append(box)
                            valid_parts.append(part_name)
                else:
                    box = boxes[gpt_part_idx]
                    # find part name in kp_data
                    kp_x, kp_y = kp_data[kp_key][part_name]
                    if kp_x != -1 and kp_y != -1: # visible
                        if check_point_in_box(kp_x*w/new_w, kp_y*h/new_h, box):
                            valid_boxes.append(box)
                            valid_parts.append(part_name)

            # imshow
            

            # visualize all kp
            # for kp_name in kp_data[kp_key]:
            #     x, y = kp_data[kp_key][kp_name]
            #     x, y = int(x*w/new_w), int(y*h/new_h)
            #     # cv2.circle(img, (x,y), radius=0, color=(0, 255, 0), thickness=10)
            #     k = part_label_dict.index(kp_name)
            #     plt.scatter(x, y, label=kp_name, s=50, edgecolors='black', c=colors_dict[k])
            # # visualize all boxes
            # for box, part in zip(boxes, chatgpt_parts):
            #     x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
            #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness) # blue
            #     cv2.putText(img, part, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # visualize valid boxes
            for box, part in zip(valid_boxes, valid_parts):
                x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness) # red
                cv2.putText(img, part, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            imgplot = plt.imshow(img)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()
            
# %%
