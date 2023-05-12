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
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}
sim_species_page = 'https://www.allaboutbirds.org/guide/Dark-eyed_Junco/species-compare'

#settings for folders
RAWFOLDER = 'allaboutbirds/'

#%%
def get_and_store_image(url: str, path: str):
    '''Obtain an image fm the world wide web and store it in the path specified'''
    response = requests.get(url, headers=HEADERS, stream=True)
    if response.status_code != 200:
        print(f"Download error {response.status_code} {url}")
    with open(path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response  
    
# %%
def show_random_img_from_folder(folder: str):
    '''Show a random image from the given folder'''
    
    filelist = [f for f in listdir(folder) if isfile(join(folder, f)) and '.jpg'in f]
    imglocation = random.choice(filelist)
    
    img=mpimg.imread(folder+'/'+imglocation)
    imgplot = plt.imshow(img)
    plt.show()
    
    print(imglocation)

# %%
def get_photoid_list(species: str) -> list:
    '''Convenience function to return a list of (already-scraped) photo ids for a given species'''
    ids = []
    for r, d, f in os.walk(RAWFOLDER+species):
        for file in f:
            ids.append(file.split('.')[0])   
    
    return ids
# %%
def species_compare_scraper(species: str, url: str = None):
    if url is None:
        # construct the url with the specified species
        URL = f'https://www.allaboutbirds.org/guide/{species}/species-compare'
    else:
        species = url.split('/')[-1]
        URL = url + 'species-compare'

    try:
        #make folders if they don't yet exist
        if not os.path.exists(RAWFOLDER+'/'+species):
            os.makedirs(RAWFOLDER+'/'+species)
        #make folders if they don't yet exist
        if not os.path.exists(RAWFOLDER+'/'+species+'/species_compare/'):
            os.makedirs(RAWFOLDER+'/'+species+'/species_compare/')
        # fetch the url and content
        if os.path.isfile(RAWFOLDER+'/'+species+'/species-compare.html'):
            print(f'The page for {species} is already there!!!')
            with open(RAWFOLDER+'/'+species+'/species-compare.html', 'rb') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        else:
            page = requests.get(URL, headers=HEADERS)
            time.sleep(1)
            with open(RAWFOLDER+'/'+species+'/species-compare.html', 'wb+') as f:
                f.write(page.content)
            soup = BeautifulSoup(page.content, 'html.parser')

        # -----Get the birds types (images, and text annotations)-----
        similarbird_tags = soup.find_all("div", {"class":"similar-species"})
        
        similarbird_dict = {}
        for child in similarbird_tags:
            bird_name_tags = child.find_all("h3")
            annotation_tags = child.find_all("div", {"class":"annotation-txt"})
            img_tags = child.find_all("img")
            
            for child1, child2, child3 in zip(bird_name_tags, annotation_tags, img_tags):
                img_links = child3.get('data-interchange') # string of list
                # Converting string to list
                img_links = img_links.replace('[',"")
                img_links = img_links.replace(']',"").split(',')
                img_links = [link for link in img_links if "http" in link] 
                img_links = list(set(img_links))
                
                bird_name = child1.get_text()
                bird_type = child2.find('h5').get_text()
                bird_desc = child2.find('p').get_text()
                bird_desc = " ".join(bird_desc.split()) # remove duplicate spaces and tabs, newlines
                
                if bird_name not in similarbird_dict:
                    similarbird_dict[bird_name] = {}
                similarbird_dict[bird_name][bird_type] = bird_desc

                # save images
                #make folders if they don't yet exist
                if not os.path.exists(RAWFOLDER+'/'+species+'/species_compare/'+bird_name):
                    os.makedirs(RAWFOLDER+'/'+species+'/species_compare/'+bird_name)
                for link in img_links:
                    filename = link.split('/')[6]
                    path = RAWFOLDER+'/'+species+'/species_compare'+'/'+bird_name+"/"+filename
                    get_and_store_image(link, path)
                    time.sleep(1)
            
        # save the descriptions
        json_object = json.dumps(similarbird_dict, indent=4)
        with open(RAWFOLDER+'/'+species+'/species_compare'+"/similarbird_dict.json", 'w') as f:
            f.write(json_object)
        
            

    except Exception as e:
        print(str(e))     

# %%
species_compare_scraper('Rock_Pigeon')
            
#%%
show_random_img_from_folder(RAWFOLDER+'/Dark-eyed_Junco')

# %%
species = ['Botteris_Sparrow', 'koolmees']
print(f'There are {len(species)} birds on the scraping list.')

# %%
for specy in species:
    species_compare_scraper(specy)
    time.sleep(1)