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

#%%
#settings for folders
RAWFOLDER = 'allaboutbirds_pages_2/'
#make folders if they don't yet exist
if not os.path.exists(RAWFOLDER):
    os.makedirs(RAWFOLDER)

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
def get_photoid_list(species: str) -> list:
    '''Convenience function to return a list of (already-scraped) photo ids for a given species'''
    ids = []
    for r, d, f in os.walk(RAWFOLDER+species):
        for file in f:
            ids.append(file.split('.')[0])   
    
    return ids

# %%
# scrape id and species compare pages
def page_scraper(species, url):
    if url is None:
        # construct the url with the specified species
        id_url = f'https://www.allaboutbirds.org/guide/{species}/id'
        species_compare_url = f'https://www.allaboutbirds.org/guide/{species}/species-compare'
    else:
        id_url = url + '/id'
        species_compare_url = url + '/species-compare'

    #make folders if they don't yet exist
    if not os.path.exists(RAWFOLDER+'/'+species):
        os.makedirs(RAWFOLDER+'/'+species)

    is_id_failed = False
    is_species_compare_failed = False
    # fetch the ID url
    if os.path.isfile(RAWFOLDER+'/'+species+'/id.html'):
        print(f'The ID page for {species} is already there!!!')
    else:
        print(f'Scraping ID page {id_url}')
        id_page = requests.get(id_url, headers=HEADERS)
        if id_page.status_code != 200:
            print(f"Download error {id_page.status_code} {url}")
            is_id_failed = True
        else:
            with open(RAWFOLDER+'/'+species+'/id.html', 'wb+') as f:
                f.write(id_page.content)
            time.sleep(1)

    # fetch the species compare url
    if os.path.isfile(RAWFOLDER+'/'+species+'/species_compare.html'):
        print(f'The scpecies_compare page for {species} is already there!!!')
    else:
        print(f'Scraping Species compare page {id_url}')
        species_compare_page = requests.get(species_compare_url, headers=HEADERS)
        if species_compare_page.status_code != 200:
            print(f"Download error {species_compare_page.status_code} {url}")
            is_species_compare_failed = True
        else:
            with open(RAWFOLDER+'/'+species+'/species_compare.html', 'wb+') as f:
                f.write(species_compare_page.content)
            time.sleep(1)

    return is_id_failed, is_species_compare_failed

# %%
# %%
df = pd.read_csv("final_nabirds_cub_search_links.csv")
auto_urls = df["Sites"].values.tolist()
manual_urls = df["Check URL manually"].values.tolist()
class_names = df['Class names'].values.tolist()
print(len(auto_urls), len(manual_urls))

for i, (auto_url, manual_url) in enumerate(zip(auto_urls, manual_urls)):
    if manual_url != manual_url: # check nan
        manual_urls[i] = auto_url
        
#%%
import shutil
num_id_failed = 0
num_species_compare_failed = 0
num_non_links = 0

bird_names = []
for i, url in enumerate(manual_urls):
    if url == 'x':
        num_non_links+=1
        continue
    
    if url[-1] == '/':
        url = url[:-1]
    bird_name = url.split('/')[-1]
    bird_names.append(bird_name)

    if not os.path.exists(RAWFOLDER+'/'+bird_name):
        print(bird_name)

    is_id_failed, is_species_compare_failed = page_scraper(bird_name, url)

    num_id_failed += is_id_failed
    num_species_compare_failed+=is_species_compare_failed

print(len(manual_urls), len(os.listdir(RAWFOLDER)))
print(f"There are {num_non_links} non-links")
print(f"ID failed: {num_id_failed}, Specices compare failed: {num_species_compare_failed}")

# %%
# -----------SCRAPING THE CONTENT---------------- #
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
            page = requests.get(URL, headers=HEADERS)
            time.sleep(1)
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
            time.sleep(1)

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
                    if child4.find('h3'):
                        type_name = child4.find('h3').get_text()
                    else:
                        type_name = "Common"        
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
                    time.sleep(1)

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
            metadata = {"Size": {'link': 'abc', 'description':[size_tag_1, size_tag_2, size_tag_3, size_tag_4]}, \
                        "Color": {'link': 'abc', 'description':color_tag}, \
                        "Behavior": {'link':'abc', 'description':behavior_tag}, \
                        "Habitat": {'link': 'abc', 'description':habitat_tag}, \
                        "Regional_Difference": {'link':'abc', 'description':regional_tag}}
            
        else: # in case there is no regional tag
            metadata = {"Size": [size_tag_1, size_tag_2, size_tag_3, size_tag_4], \
                        "Color": color_tag, "Behavior": behavior_tag, \
                        "Habitat": habitat_tag}
        #store metadata to flat file
        store_meta(species, metadata)
    except Exception as e:
        print(str(e))      

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
id_scraper('Rock_Pigeon')


# %%
# species = ['Botteris_Sparrow', 'Rosss_Goose', 'Rock_Pigeon', 'Scaled_Quail']
# print(f'There are {len(species)} birds on the scraping list.')

# for specy in species:
#     id_scraper(specy)


# %%
