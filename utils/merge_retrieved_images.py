# %%
import os
import cv2
import numpy as np
import math
import pandas as pd
import json
# %%
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_thickness = 1

def merge_images(query_and_image_paths, class_name="EMPTY"):
    """
    query_and_image_paths: json file containing query and paths of images that need to be concatenated
    class_name: class of input images
    return np image
    """

    query_images = []
    image_names = []

    H, W = 224, 224
    f = open(query_and_image_paths)
    data = json.load(f)

    for query, image_paths in data.items():
        images = []
        for image_path in image_paths:
            #load image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (W, H))
            # get file name
            image_name = image_path.split('/')[-1]
            text_size = cv2.getTextSize(image_name, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size
            text_x = int((W - text_width) / 2)
            text_y = H + text_height + 10

            white_region_height = text_y + text_height 
            white_region_width = W
            white_region = np.ones((white_region_height, white_region_width, 3), dtype=np.uint8) * 255
            white_region[:H, :W, :] = image
            cv2.putText(white_region, image_name, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
            #
            images.append(white_region)

        len_images = len(images)
        columns, rows = 5, math.ceil(len_images/5) # predefined number of columns is 5

        extended_images = []
        for i in range(rows):
            row_images = []
            for j in range(columns):
                mapped_idx = 5*i + j
                if mapped_idx < len_images:
                    row_images.append(images[mapped_idx])
                else:
                    row_images.append(np.zeros((H, W, 3)))
            
            row_images = np.array(row_images)
            row_image = np.concatenate(row_images, axis=1)
            extended_images.append(row_image)
        
        extended_images = np.array(extended_images)
        extended_image = np.concatenate(extended_images, axis=0)
        # add query
        HH, WW = extended_image.shape[:2]
        text_size = cv2.getTextSize(query, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        text_x = int((WW - text_width) / 2)
        text_y = HH + text_height + 10

        white_region_height = text_y + text_height 
        white_region_width = WW
        white_region = np.ones((white_region_height, white_region_width, 3), dtype=np.uint8) * 255
        white_region[:HH, :WW, :] = extended_image
        
        cv2.putText(white_region, query, (text_x-90, text_y), font, 0.4, (0, 0, 0), font_thickness, cv2.LINE_AA)
        query_images.append(white_region)
    
    query_image = np.concatenate(np.array(query_images), axis=0)

    return query_image

# %%
folder_paths = ["/home/tin/projects/reasoning/plain_clip/retrieved_cub_",\
               "/home/tin/projects/reasoning/plain_clip/retrieved_cub_inat21", \
               "/home/tin/projects/reasoning/plain_clip/retrieved_cub_nabirds",\
                "/home/tin/projects/reasoning/plain_clip/retrieved_cub_nabirds_inat21"
]

# %%
path = folder_paths[0]

save_folder = f"./{path.split('/')[-1]}"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

label_folders = os.listdir(path)
for cls_name in label_folders:
    json_path = f"{path}/{cls_name}/query.json"
    image = merge_images(json_path, cls_name)
    cv2.imwrite(f"{save_folder}/{cls_name}.jpg", image)
# %%
