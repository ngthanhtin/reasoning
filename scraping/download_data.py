# %%
import requests
from bs4 import BeautifulSoup
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import random
import cv2
import numpy as np
import pandas as pd
import time
import json
import multiprocessing as mp

# webpages
HEADERS = {'accept': '"text/html', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}

ID_FOLDER = 'allaboutbirds_ids/'
SPECIES_COMPARE_FOLDER = 'allaboutbirds_species_compares/'

SAVE_DATA_FOLDER = 'allaboutbirds_data/'
SAVE_ID_DATA_FOLDER = f'{SAVE_DATA_FOLDER}/id_data/'
SAVE_SC_DATA_FOLDER = f'{SAVE_DATA_FOLDER}/species_compare_data/'

#make folders if they don't yet exist
if not os.path.exists(SAVE_DATA_FOLDER):
    os.makedirs(SAVE_DATA_FOLDER)
if not os.path.exists(SAVE_ID_DATA_FOLDER):
    os.makedirs(SAVE_ID_DATA_FOLDER)
if not os.path.exists(SAVE_SC_DATA_FOLDER):
    os.makedirs(SAVE_SC_DATA_FOLDER)

#%%
def get_and_store_image(url: str, path: str):
    '''Obtain an image fm the world wide web and store it in the path specified'''
    response = requests.get(url, headers=HEADERS, stream=True)
    if response.status_code != 200:
        print(f"Download error {response.status_code} {url}")
        del response
        return 0
    with open(path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return 1  

def get_photo_list(path):
    return os.listdir(path)
    
# %%
# ----------DOWNLOAD ID IMAGES----------------#
def download_meta_images(meta_path, metadata_data_folder):
    
    photo_list = get_photo_list(metadata_data_folder)

    # read json
    f = open(meta_path)
    metadata = json.load(f)
    num_failure = 0
    for k in metadata:
        if 'link' in metadata[k]:
            links = metadata[k]['link']
            link = next((item for item in links if '720' in item), None)
            if not link: # there is no 720 px images, or it is a video link
                link = links if isinstance(links, str) else links[0]
            if 'video' in link: # for now, not download video
                continue
            
            filename = f"{k}.jpg"
            if filename in photo_list:
                continue
            success = get_and_store_image(link, os.path.join(metadata_data_folder, filename))
            if success == 0:
                print(meta_path)
                num_failure += 1
            time.sleep(1)

    return num_failure

def download_bird_type_images(bird_type_path, bird_type_data_folder):
    photo_list = get_photo_list(bird_type_data_folder)

    # read json
    f = open(bird_type_path)
    bird_type_data = json.load(f)
    num_failure = 0
    for k in bird_type_data:
        if 'link' in bird_type_data[k]:
            list_links = bird_type_data[k]['link']
            for i, links in enumerate(list_links):
                link = next((item for item in links if '720' in item), None)
                if not link: # there is no 720 px images, or it is a video link
                    link = links if isinstance(links, str) else links[0]
                if 'video' in link: # for now, not download video
                    continue
            
                filename = f"{k}_{i}.jpg"
                if filename in photo_list:
                    continue
                success = get_and_store_image(link, os.path.join(bird_type_data_folder, filename))
                if success == 0:
                    print(bird_type_path)
                    num_failure += 1
                time.sleep(1)

    return num_failure

# %%
# ------ DOWNLOAD META DATA DATA--------- #
bird_names = os.listdir(ID_FOLDER)

for bird_name in bird_names:
    id_path = os.path.join(ID_FOLDER, bird_name)
    meta_path = f"{id_path}/meta.json"
    # create folders
    if not os.path.exists(f"{SAVE_ID_DATA_FOLDER}/{bird_name}"):
        os.makedirs(f"{SAVE_ID_DATA_FOLDER}/{bird_name}")
    metadata_data_folder = f"{SAVE_ID_DATA_FOLDER}/{bird_name}/metadata_data"
    
    if not os.path.exists(metadata_data_folder):
        os.makedirs(metadata_data_folder)
    
    # -------------
    download_meta_images(meta_path, metadata_data_folder)

#%%
# ------ DOWNLOAD BIRD TYPE DATA -------- #
bird_names = os.listdir(ID_FOLDER)

for bird_name in bird_names:
    bird_path = os.path.join(ID_FOLDER, bird_name)
    bird_type_path = f"{bird_path}/bird_type_dict.json"
    # create folders
    if not os.path.exists(f"{SAVE_ID_DATA_FOLDER}/{bird_name}"):
        os.makedirs(f"{SAVE_ID_DATA_FOLDER}/{bird_name}")
    bird_type_data_folder = f"{SAVE_ID_DATA_FOLDER}/{bird_name}/bird_type_data"

    if not os.path.exists(bird_type_data_folder):
        os.makedirs(bird_type_data_folder)

    # ---------------
    download_bird_type_images(bird_type_path, bird_type_data_folder)



#%%
#-----------DOWNLOAD SPECIES COMPARE IMAGES-------------#
def download_similar_bird_images(similar_bird_path, similar_bird_data_folder):
    photo_list = get_photo_list(similar_bird_data_folder)

    # read json
    f = open(similar_bird_path)
    similar_bird_data = json.load(f)
    num_failure = 0
    for similar_bird_name in similar_bird_data:
        for bird_type in similar_bird_data[similar_bird_name]:
            links = similar_bird_data[similar_bird_name][bird_type]['link']
            link = next((item for item in links if '720' in item), None)
            if not link: # there is no 720 px images, or it is a video link
                link = links if isinstance(links, str) else links[0]
            if 'video' in link: # for now, not download video
                continue
            
            filename = f"{similar_bird_name}_{bird_type}.jpg"
            if filename in photo_list:
                continue
            success = get_and_store_image(link, os.path.join(similar_bird_data_folder, filename))
            if success == 0:
                print(similar_bird_path)
                num_failure += 1
            time.sleep(1)

    return num_failure

# %%
bird_names = os.listdir(SPECIES_COMPARE_FOLDER)

for bird_name in bird_names:
    bird_path = os.path.join(SPECIES_COMPARE_FOLDER, bird_name)
    sc_path = f"{bird_path}/similarbird_dict.json"
    # create folders
    similar_bird_data_folder = f"{SAVE_SC_DATA_FOLDER}/{bird_name}"
    if not os.path.exists(similar_bird_data_folder):
        os.makedirs(similar_bird_data_folder)

    # ---------------
    download_similar_bird_images(sc_path, similar_bird_data_folder)