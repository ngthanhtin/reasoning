import copy
import os
import textwrap

import numpy
import numpy as np
import PIL
import torch
from matplotlib import cm
from mmdet.core.visualization import imshow_det_bboxes
from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm import tqdm

import mmcv


######## MMdetection drawing segmentation/boxes ########
def draw_segments(
                img,
                results,
                text_list,
                labels,
                scores=None,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=None,
                text_bgcolor='gray',
                mask_color=None,
                thickness=3,
                font_size=13,
                win_name='',
                wait_time=0,
                cmap_set: list[str] or None = None,
                emb_type = 'mask2former'
                ):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        results (Tensor or tuple): The results to draw over `img` bbox_result or (bbox_result, segm_result).
        text_list (list[str]): The text to write over `img`
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
            The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
            The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
            Color of masks. The tuple of color should be in BGR order.
            Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    # pan_results = result['pan_results']
    # keep objects ahead

    if emb_type.startswith("mask2former"):
        # keep objects ahead
        # ids = np.unique(results)[::-1]
        # segms = (results[None] == ids[:, None, None])
        segms = results[0].numpy()
    elif emb_type == 'fasterrcnn':
        segms = results[0]
    elif emb_type == 'union':
        segms = results[0]

    if labels.shape[0] != segms.shape[0]:
        print("SIZE MISMATCH")
        return img

    if mask_color is None:
        mask_color = get_pre_define_colors(len(text_list), cmap_set)
        text_color = get_pre_define_colors(len(text_list), cmap_set)

    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        scores=scores,
        labels=labels,
        segms=segms,
        class_names=text_list,
        # score_thr=score_thr, #* for bounding boxes only
        bbox_color=bbox_color,
        text_color=text_color,
        text_bgcolor=text_bgcolor,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False,
        wait_time=wait_time,
        out_file=None)

    # cat_img = Drawer.concat([Image.fromarray(img), Image.fromarray(pred_img)], horizontal=True)
    return numpy.asarray(img)
######## MMdetection drawing segmentation/boxes ########


def get_pre_define_colors(num_classes, cmap_set: list[str] = None, in_rgb: bool = True):
    if cmap_set is None:
        cmap_set = ['tab20', 'tab20b', 'tab20c']

    colors = []
    for cmap in cmap_set:
        colors.extend(list(cm.get_cmap(cmap).colors))

    if in_rgb:
        colors = [tuple(int(x * 255) for x in color) for color in colors]

    if num_classes > len(colors):
        print(f"WARNING: {num_classes} classes are requested, but only {len(colors)} colors are available. Predefined colors will be reused.")
        colors *= num_classes // len(colors) + 1

    return colors


def draw_text(img, text_list, text_size=14, text_color=None, inside=False):
    try:
        ft = ImageFont.truetype("/home/tin/xclip/src/arial.ttf", text_size)
    except Exception:
        ft = ImageFont.truetype("arial.ttf", text_size)
    # else:
    #     raise FileNotFoundError("No font file found")

    # ft = ImageFont.truetype("/Library/fonts/Arial.ttf", 14)
    img_width, img_height = img.size
    reformated_lines = []
    text_lines_height = []

    scale = 2.5 if img_width < 700 or img_height < 700 else 2
    new_img_width = int(img_width * scale) if img_width <= img_height else img_width
    new_img_height = int(img_height * scale) if img_height < img_width else img_height
    compared_width = new_img_width - img_width if new_img_width > img_width else img_width

    for line in text_list:
        line_width, line_height = ft.getsize(line)
        if line_width > compared_width:  # wrap text to multiple lines
            character_width = line_width/(len(line))
            characters_per_line = round(compared_width/character_width) - 2
            sperate_lines = textwrap.wrap(line, characters_per_line)
            line = '\n'.join(sperate_lines)
            text_lines_height.append(len(sperate_lines))
        else:
            text_lines_height.append(1)
        reformated_lines.append(line)

    total_lines = sum(text_lines_height)
    if inside:
        text_img = Image.new('RGB', (img_width, line_height*total_lines+10), (255, 255, 255))
        img.paste(text_img, (0, 0, img_width, line_height*total_lines+10))
        new_img = img
        draw = ImageDraw.Draw(new_img)
        y_text = 0
    else:
        # new_img = Image.new('RGB', (img_width, img_height+line_height*total_lines+10), (255, 255, 255))
        new_img = Image.new('RGB', (new_img_width, new_img_height), (255, 255, 255))
        new_img.paste(img, (0, 0, img_width, img_height))
        draw = ImageDraw.Draw(new_img)
        x_text = img_width + 3 if img_width <= img_height else 3
        y_text = img_height + 3 if img_height < img_width else 3

    for idx, (line, color) in enumerate(zip(reformated_lines, text_color)):
        draw.text((x_text, y_text), line, color, font=ft)
        y_text += (line_height+2) * text_lines_height[idx]

    return new_img


def draw_box(img, box, color='red', width=3):
    new_img = copy.deepcopy(img)
    draw = ImageDraw.Draw(new_img, mode="RGBA")
    draw.rectangle(box, outline=color, width=width)
    return new_img


def generate_text_img(text, text_color, mode='RGB', text_size = 12, img_width=False):
    ft = ImageFont.truetype("arial.ttf", text_size)
    w, h = ft.getsize(text)
    # compute # of lines
    # lines = math.ceil(img_width / width) +     
    height = h
    if len(mode) == 1: # L, 1
        background = (255)
        color = (0)
    if len(mode) == 3: # RGB
        background = (255, 255, 255)
        color = (0,0,0)
    if len(mode) == 4: # RGBA, CMYK
        background = (255, 255, 255, 255)
        color = (0,0,0,0)
    if img_width:
        textImage = Image.new(mode, (img_width, height), background)
    else:
        textImage = Image.new(mode, (w, height), background)
    draw = ImageDraw.Draw(textImage)  

    # ipdb.set_trace()
    # tx_w, tx_h = ft.getsize(text)
    draw.text((5, 0), text, text_color, font=ft)

    return textImage


"""
Concatenate images with imagick
Input:
# dir_list: A list of dirs, each dir must have same amount of images and same name
# out_dir : output folder 
Output:
A set concatenate images
"""
def concat_imgs(dir_list, out_dir='visualization_samples/clip_context/s_iMs_M_montage'):
    os.makedirs(out_dir, exist_ok=True)
    img_list = os.listdir(dir_list[0])
    num_of_dirs = len(dir_list)
    montage_file_list = []

    for img_name in img_list:
        montage_files = ''.join(f'{dir_list[i]}/{img_name} ' for i in range(num_of_dirs))
        montage_file_list.append(montage_files)

    for files, img_name in tqdm(zip(montage_file_list, img_list), total=len(img_list)):
        os.system(f'montage -quiet {files}-tile {num_of_dirs}x1 -geometry +0+0 {out_dir}/{img_name}')


class Drawer:
    @staticmethod
    def draw_boxes(image, boxes, colors, tags=None, alpha=0.8, width=1, text_size=12, loc='above'):
        if tags is not None:
            try:
                font = ImageFont.truetype("/home/tin/xclip/src/arial.ttf", text_size)
            except Exception:
                font = ImageFont.truetype("arial.ttf", text_size)
            if len(boxes) != len(tags):
                raise ValueError('boxes and tags must have same length')

        for idx, box in enumerate(boxes):
            # If there are duplicated boxes, slightly adjust x and y for better visualization
            if boxes.count(box) > 1:
                box[0] += np.random.randint(-10, 10) * 0.1
                box[1] += np.random.randint(-10, 10) * 0.1

            color_rgba = colors[idx] + (int(alpha * 255),)
            image = draw_box(image, box, color=color_rgba, width=width)

            if tags is not None:
                tag = tags[idx]
                draw = ImageDraw.Draw(image, 'RGBA')
                tag_width, tag_height = font.getmask(tag).size

                if loc == 'above':
                    textbb_loc = [box[0], box[1] - tag_height, box[0] + tag_width, box[1]]
                    text_loc = box[0], box[1] - tag_height
                else:
                    textbb_loc = [box[0], box[1], box[0] + tag_width, box[1] + tag_height]
                    text_loc = box[0], box[1]

                draw.rectangle(textbb_loc, fill=color_rgba)
                draw.text(text_loc, tag, fill='white', font=font)

        return image

    @staticmethod
    def draw_text(image, text_list): # draw text as extra box in image
        image = draw_text(image, text_list)
        return image
        
    @staticmethod
    def concat(target_image_list: PIL.Image, horizontal: bool = True) -> PIL.Image:
        widths, heights = [], []
        for image in target_image_list:
            width, height = image.size
            widths.append(width)
            heights.append(height)

        total_width, max_width = sum(widths), max(widths)
        total_height, max_height = sum(heights), max(heights)

        # num_imgs = len(target_image_list)
        cat_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255)) if horizontal else Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
        for idx, img in enumerate(target_image_list):
            if horizontal:
                cat_img.paste(img, (sum(widths[:idx]), 0))
            else:
                cat_img.paste(img, (0, sum(heights[:idx])))
        return cat_img
    
    @staticmethod
    def paste_patch_to_image(box, img_patch, org_image):
        image = copy.deepcopy(org_image)
        image_width, image_height = image.size
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, image_width)
        y2 = min(y2, image_height)
        img_patch = img_patch.resize((x2-x1, y2-y1))
        image.paste(img_patch, (x1, y1))
        return image
    
    @staticmethod
    def concat_imgs_in_folders(dir1, dir2, dir3=None, out_dir=None, horizontal=True):
        os.makedirs(out_dir, exist_ok=True), 
        img_list = os.listdir(dir1)
        for img_name in img_list:
            if img_name.endswith(".json"):
                continue

            img1 = Image.open(f'{dir1}/{img_name}')
            img2 = Image.open(f'{dir2}/{img_name}')
            if dir3 is not None:
                img3 = Image.open(f'{dir3}/{img_name}')
                cat_img = Drawer.concat([img1, img2, img3], horizontal=horizontal)
            else:
                cat_img = Drawer.concat([img1, img2], horizontal=horizontal)
            cat_img.save(f'{out_dir}/{img_name}', quality=100, subsampling=0)
    
    @staticmethod
    def check_name_matching(folder1, folder2):
        img_list1 = os.listdir(folder1)
        img_list2 = os.listdir(folder2)
        diff = set(img_list1) - set(img_list2)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder2}')
            print(diff)
        diff = set(img_list2) - set(img_list1)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder1}')
            print(diff)
        return len(diff) == 0

