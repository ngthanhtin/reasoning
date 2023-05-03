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
id_page = 'https://www.allaboutbirds.org/guide/Dark-eyed_Junco/id'
gallery_page = 'https://www.allaboutbirds.org/guide/Dark-eyed_Junco/photo-gallery'
sim_species_page = 'https://www.allaboutbirds.org/guide/Dark-eyed_Junco/species-compare'

#settings for folders
RAWFOLDER = 'allaboutbirds/'

species = [{
    'name': 'pimpelmees',
    'id': '161'
}, {
    'name': 'koolmees',
    'id': '140'
}]

print(f'There are {len(species)} birds on the scraping list.')

#%%
def get_and_store_image(url: str, path: str):
    '''Obtain an image fm the world wide web and store it in the path specified'''
    response = requests.get(url, stream=True)
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
def id_scraper(species: str):
    # construct the url
    URL = f'https://www.allaboutbirds.org/guide/{species}/id'

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
            page = requests.get(URL)
            with open(RAWFOLDER+'/'+species+'/id.html', 'wb+') as f:
                f.write(page.content)
            soup = BeautifulSoup(page.content, 'html.parser')
        
        # -----Get a representative image for this species-----
        photo_tags = soup.findAll('img', {"alt":"Dark-eyed Junco"})
        # get the photoids we already have scraped from - the links change
        photoids = get_photoid_list(species)
        for photo_tag in photo_tags:
            image_urls = photo_tag.get('data-interchange') # string of list
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
                    with open(RAWFOLDER+'/'+species+'/'+type_name+"/description.txt", 'w') as f:
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
        regional_tag = text_tags.find("p")

        metadata = {"Size": [size_tag_1, size_tag_2, size_tag_3, size_tag_4], \
                    "Color": color_tag, "Behavior": behavior_tag, \
                    "Habitat": habitat_tag, \
                    "Regional_Difference": regional_tag}
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
id_scraper('Dark-eyed_Junco')
            
#%%
show_random_img_from_folder(RAWFOLDER+'/Dark-eyed_Junco')
# %%
# for s in species[30:]:
#     bird_scraper(s['name'], s['id'])
    
#     # show a random photo
#     print(str(s['name'])+':')
#     show_random_img_from_folder(RAWFOLDER+'/'+s['name'])

# %%
