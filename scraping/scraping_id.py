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
species = "Dark-eyed_Junco"
id_page = f'https://www.allaboutbirds.org/guide/{species}/id'

#settings for folders
RAWFOLDER = 'allaboutbirds/'

#%%
def get_and_store_image(url: str, path: str):
    '''Obtain an image fm the world wide web and store it in the path specified'''
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Download error {url}")
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
def id_scraper(species: str, url: str = None):
    if url is None:
        # construct the url with the specified species
        URL = f'https://www.allaboutbirds.org/guide/{species}/id'
    else:
        species = url.split('/')[-1]
        URL = url + 'id'

    try:
        #make folders if they don't yet exist
        if not os.path.exists(RAWFOLDER+'/'+species):
            os.makedirs(RAWFOLDER+'/'+species)
        # fetch the url and content
        if os.path.isfile(RAWFOLDER+'/'+species+'/id.html'):
            print(f'The page for {species} is already there!!!')
            with open(RAWFOLDER+'/'+species+'/id.html', 'rb') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        else:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}
            page = requests.get(URL, headers=headers)
            with open(RAWFOLDER+'/'+species+'/id.html', 'wb+') as f:
                f.write(page.content)
            soup = BeautifulSoup(page.content, 'html.parser')
        
        # -----Get a representative image for this species-----
        photo_tags = soup.find('aside', {"aria-label":"Shape Media"}).find("img")
        # get the photoids we already have scraped from - the links change
        photoids = get_photoid_list(species)
        image_urls = photo_tags.get('data-interchange') # string of list
        # Converting string to list
        image_urls = image_urls.replace('[',"")
        image_urls = image_urls.replace(']',"").split(',')
        image_urls = [url for url in image_urls if "http" in url]
        image_urls = list(set(image_urls))
        image_url = image_urls[1]
        
        filename = f"common_{species}.jpg"
        #check if we have encountered this photo before- will be substantially slower with large n
        if filename.split('.')[0] not in photoids: 
            path = RAWFOLDER+'/'+species+'/'+filename
            get_and_store_image(image_url, path)

        # -----Get the birds types (images, and text annotations)-----
        birdtype_tags=soup.find("section",{"aria-labelledby":"photos-heading"})
        birdtype_tags = birdtype_tags.find("div", {"class":"slider slick-3"})
        
        children = birdtype_tags.findChildren("div" , recursive=False)
        for child1 in children:
            child2 = child1.findChildren("a", recursive=False)
            for child3 in child2:
                img_tag = child3.findChildren("img", recursive=False)
                for child4 in img_tag:
                    img_links = child4.get('data-interchange') # string of list
                    # Converting string to list
                    img_links = img_links.replace('[',"")
                    img_links = img_links.replace(']',"").split(',')
                    img_links = [link for link in img_links if "http" in link] 
                    img_links = list(set(img_links))

                annotation_tag = child3.findChildren("div",{"class":"annotation-txt"})
                for child4 in annotation_tag:
                    type_name = child4.find('h3').get_text()
                    description = child4.find('p').get_text()
                    description = " ".join(description.split()) # remove duplicate spaces and tabs, newlines
                    type_name = type_name.replace('/',' and ') # if any "/" in the string
                #make folders if they don't yet exist
                if not os.path.exists(RAWFOLDER+'/'+species+'/'+type_name):
                    os.makedirs(RAWFOLDER+'/'+species+'/'+type_name)
                with open(RAWFOLDER+'/'+species+'/'+type_name+"/description.txt", 'a') as f:
                    f.write(description)
                for link in img_links:
                    filename = link.split('/')[6]
                    path = RAWFOLDER+'/'+species+'/'+type_name+'/'+filename
                    get_and_store_image(link, path)

        # -----Get the text size & shape-----
        text_tags=soup.find('article', {"aria-label":"Size & Shape"})
        size_tag_1 = text_tags.find("p")
        size_tag_2 = text_tags.find("div").find("p")
        size_tag_3 = text_tags.find("div").find("span")
        size_tag_4 = text_tags.find("div").find("ul").find_all('li')

        # -----Get the color pattern-----
        text_tags=soup.find('article', {"aria-label":"Color Pattern"})
        color_tag = text_tags.find("p")
        # -----Get the behaviour-----
        text_tags=soup.find('article', {"aria-label":"Behavior"})
        behavior_tag = text_tags.find("p")
        # -----Get the habitat-----
        text_tags=soup.find('article', {"aria-label":"Habitat"})
        habitat_tag = text_tags.find("p")
        # -----Get the regional differences-----
        text_tags=soup.find('article', {"aria-label":"Regional Differences"})
        if text_tags:
            regional_tag = text_tags.find("p")
            metadata = {"Size": [size_tag_1, size_tag_2, size_tag_3, size_tag_4], \
                        "Color": color_tag, "Behavior": behavior_tag, \
                        "Habitat": habitat_tag, \
                        "Regional_Difference": regional_tag}
            
        else: # in case there is no regional tag
            metadata = {"Size": [size_tag_1, size_tag_2, size_tag_3, size_tag_4], \
                        "Color": color_tag, "Behavior": behavior_tag, \
                        "Habitat": habitat_tag}
        #store metadata to flat file
        store_meta(species, metadata)
    except Exception as e:
        print(str(e))

# %%
def store_meta(species: str, meta: list):
    meta_dict = {}
    for k in meta.keys():
        if k == 'Size':
            meta_dict['Size'] = []
            for i in range(len(meta[k])):
                if i == 3: # measurement
                    for idx, ele in enumerate(meta[k][i]):
                        if idx == 1:
                            continue
                        if idx == 0: # sex
                            meta_dict['Size'].append("Sex: "+ ele.text.strip())
                        else:
                            meta_dict['Size'].append(ele.text.strip())
                else:
                    for idx, ele in enumerate(meta[k][i]):
                        meta_dict['Size'].append(ele.text.strip())
        else:
            meta_dict[k] = []
            for idx, ele in enumerate(meta[k]):
                    meta_dict[k].append(ele.text.strip())
    
    json_object = json.dumps(meta_dict, indent=4)
    with open(f"{RAWFOLDER}/{species}/meta.json", "w") as outfile:
        outfile.write(json_object)

    return meta_dict        

# %%
id_scraper('Spotted_Towhee')
            
#%%
show_random_img_from_folder(RAWFOLDER+'/Dark-eyed_Junco')

# %%

species = ['Botteris_Sparrow']
print(f'There are {len(species)} birds on the scraping list.')

for specy in species:
    id_scraper(specy)
# %%
