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
BIRD_PAGE_FOLDER = 'allaboutbirds_pages/'
ID_FOLDER = 'allaboutbirds_ids/'
SPECIES_COMPARE_FOLDER = 'allaboutbirds_species_compares/'
#make folders if they don't yet exist
if not os.path.exists(BIRD_PAGE_FOLDER):
    os.makedirs(BIRD_PAGE_FOLDER)
if not os.path.exists(ID_FOLDER):
    os.makedirs(ID_FOLDER)
if not os.path.exists(SPECIES_COMPARE_FOLDER):
    os.makedirs(SPECIES_COMPARE_FOLDER)

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
    if not os.path.exists(BIRD_PAGE_FOLDER+'/'+species):
        os.makedirs(BIRD_PAGE_FOLDER+'/'+species)

    is_id_failed = False
    is_species_compare_failed = False
    # fetch the ID url
    if os.path.isfile(BIRD_PAGE_FOLDER+'/'+species+'/id.html'):
        print(f'The ID page for {species} is already there!!!')
    else:
        print(f'Scraping ID page {id_url}')
        id_page = requests.get(id_url, headers=HEADERS)
        if id_page.status_code != 200:
            print(f"Download error {id_page.status_code} {url}")
            is_id_failed = True
        else:
            with open(BIRD_PAGE_FOLDER+'/'+species+'/id.html', 'wb+') as f:
                f.write(id_page.content)
            time.sleep(1)

    # fetch the species compare url
    if os.path.isfile(BIRD_PAGE_FOLDER+'/'+species+'/species-compare.html'):
        print(f'The scpecies_compare page for {species} is already there!!!')
    else:
        print(f'Scraping Species compare page {id_url}')
        species_compare_page = requests.get(species_compare_url, headers=HEADERS)
        if species_compare_page.status_code != 200:
            print(f"Download error {species_compare_page.status_code} {url}")
            is_species_compare_failed = True
        else:
            with open(BIRD_PAGE_FOLDER+'/'+species+'/species_compare.html', 'wb+') as f:
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

    if not os.path.exists(BIRD_PAGE_FOLDER+'/'+bird_name):
        print(bird_name)

    is_id_failed, is_species_compare_failed = page_scraper(bird_name, url)

    num_id_failed += is_id_failed
    num_species_compare_failed+=is_species_compare_failed

print(len(manual_urls), len(os.listdir(BIRD_PAGE_FOLDER)))
print(f"There are {num_non_links} non-links")
print(f"ID failed: {num_id_failed}, Specices compare failed: {num_species_compare_failed}")

# %%
# -----------SCRAPING THE CONTENT---------------- #
# %%
def store_meta(species: str, meta: list):
    meta_dict = {}
    for k in meta.keys():
        if k == 'Size':
            if 'link' in meta['Size']:
                meta_dict['Size'] = {'link':meta['Size']['link'], 'description': []}
            else:
                meta_dict['Size'] = {'description': []}

            for i in range(len(meta[k]['description'])):
                if i == 3: # measurement
                    for idx, ele in enumerate(meta[k]['description'][i]):
                        if idx == 1:
                            continue
                        if idx == 0: # sex
                            meta_dict['Size']['description'].append("Sex: "+ ele.text.strip())
                        else:
                            meta_dict['Size']['description'].append(ele.text.strip())
                else:
                    for idx, ele in enumerate(meta[k]['description'][i]):
                        meta_dict['Size']['description'].append(ele.text.strip())
        else:
            if 'link' in meta[k]:
                meta_dict[k] = {'link': meta[k]['link'], 'description': []}
                for idx, ele in enumerate(meta[k]['description']):
                        meta_dict[k]['description'].append(ele.text.strip())
            else:
                meta_dict[k] = {'description': []}
                for idx, ele in enumerate(meta[k]['description']):
                        meta_dict[k]['description'].append(ele.text.strip())
    
    json_object = json.dumps(meta_dict, indent=4)
    with open(f"{ID_FOLDER}/{species}/meta.json", "w") as outfile:
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
        if not os.path.exists(ID_FOLDER+'/'+species):
            os.makedirs(ID_FOLDER+'/'+species)
        # fetch the url and content
        if os.path.isfile(BIRD_PAGE_FOLDER+'/'+species+'/id.html'):
            print(f'The page for {species} is already there!!!')
            with open(BIRD_PAGE_FOLDER+'/'+species+'/id.html', 'rb') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        else:
            page = requests.get(URL, headers=HEADERS)
            time.sleep(1)
            with open(BIRD_PAGE_FOLDER+'/'+species+'/id.html', 'wb+') as f:
                f.write(page.content)
            soup = BeautifulSoup(page.content, 'html.parser')
        
        # -----Get "Shape Media" image for this species-----
        photo_tags = soup.find('aside', {"aria-label":"Shape Media"})
        if photo_tags:
            photo_tags = photo_tags.find("img")
            shape_image_urls = photo_tags.get('data-interchange') # string of list
            # Converting string to list
            shape_image_urls = shape_image_urls.replace('[',"")
            shape_image_urls = shape_image_urls.replace(']',"").split(',')
            shape_image_urls = [url for url in shape_image_urls if "http" in url]
            shape_image_urls = list(set(shape_image_urls))
        else:
            shape_image_urls = None

        # -----Get "Color Pattern Media" image for this species-----
        photo_tags = soup.find('aside', {"aria-label":"Color Pattern Media"})
        if photo_tags:
            photo_tags = photo_tags.find("img")
            color_image_urls = photo_tags.get('data-interchange') # string of list
            # Converting string to list
            color_image_urls = color_image_urls.replace('[',"")
            color_image_urls = color_image_urls.replace(']',"").split(',')
            color_image_urls = [url for url in color_image_urls if "http" in url]
            color_image_urls = list(set(color_image_urls))
        else:
            color_image_urls = None

        # -----Get "Behavior Media" image for this species-----
        photo_tags = soup.find('aside', {"aria-label":"Behavior Media"})
        if photo_tags:
            photo_tags = photo_tags.find("iframe")
            if photo_tags:
                behavior_image_urls = photo_tags.get('src') # string of video 
            else:
                photo_tags = soup.find('aside', {"aria-label":"Behavior Media"})
                if photo_tags:
                    photo_tags = photo_tags.find("img")
                    if photo_tags:
                        behavior_image_urls = photo_tags.get('data-interchange') # string of list
                        # Converting string to list
                        behavior_image_urls = behavior_image_urls.replace('[',"")
                        behavior_image_urls = behavior_image_urls.replace(']',"").split(',')
                        behavior_image_urls = [url for url in behavior_image_urls if "http" in url]
                        behavior_image_urls = list(set(behavior_image_urls))  
                    else:
                        behavior_image_urls = None
                # else:
                #     behavior_image_urls = None
                
        # -----Get "Habitat Media" image for this species-----
        photo_tags = soup.find('aside', {"aria-label":"Habitat Media"})
        if photo_tags:
            photo_tags = photo_tags.find("img")
            if photo_tags:
                habitat_image_urls = photo_tags.get('data-interchange') # string of list
                # Converting string to list
                habitat_image_urls = habitat_image_urls.replace('[',"")
                habitat_image_urls = habitat_image_urls.replace(']',"").split(',')
                habitat_image_urls = [url for url in habitat_image_urls if "http" in url]
                habitat_image_urls = list(set(habitat_image_urls))
            else:
                photo_tags = soup.find('aside', {"aria-label":"Habitat Media"})
                photo_tags = photo_tags.find("iframe")
                if photo_tags:
                    habitat_image_urls = photo_tags.get('src') # string of video
                else:
                    habitat_image_urls = None
        else:
            habitat_image_urls = None
        
        # -----Get "Regional Differences" image for this species-----
        photo_tags = soup.find('aside', {"aria-labeledby":"regional-photos"})
        if photo_tags:
            photo_tags = photo_tags.find("img")
            region_image_urls = photo_tags.get('data-interchange') # string of list
            # Converting string to list
            region_image_urls = region_image_urls.replace('[',"")
            region_image_urls = region_image_urls.replace(']',"").split(',')
            region_image_urls = [url for url in region_image_urls if "http" in url]
            region_image_urls = list(set(region_image_urls))
        else:
            region_image_urls = None

        # -----Get the birds types (images, and text annotations)-----
        birdtype_tags=soup.find("section",{"aria-labelledby":"photos-heading"})
        birdtype_tags = birdtype_tags.find("div", {"class":"slider slick-3"})
        
        bird_type_dict = {}
        children = birdtype_tags.findChildren("div" , recursive=False)
        for child1 in children:
            child2 = child1.findChildren("a", recursive=False)
            for child3 in child2:
                img_tag = child3.findChildren("img", recursive=False)
                if len(img_tag) == 0: # there is a video tag
                    img_tag = child3.find("img")
                    img_tag = [img_tag]
                    
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
                # # make folders if they don't yet exist
                # if not os.path.exists(RAWFOLDER+'/'+species+'/'+type_name):
                #     os.makedirs(RAWFOLDER+'/'+species+'/'+type_name)
                # with open(RAWFOLDER+'/'+species+'/'+type_name+"/description.txt", 'a') as f:
                #     f.write(description)
                if not type_name in bird_type_dict:
                    bird_type_dict[type_name] = {"link": [img_links], "description": [description]}
                else:
                    bird_type_dict[type_name]["link"].append(img_links)
                    bird_type_dict[type_name]["description"].append(description)
                # for link in img_links:
                #     filename = link.split('/')[6]
                #     path = RAWFOLDER+'/'+species+'/'+type_name+'/'+filename
                    # get_and_store_image(link, path)
                    # time.sleep(1)
        
        # save the descriptions
        json_object = json.dumps(bird_type_dict, indent=4)
        with open(f"{ID_FOLDER}/{species}/bird_type_dict.json", 'w') as f:
            f.write(json_object)

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
        
        metadata = {}
        if shape_image_urls:
            metadata["Size"] = {'link': shape_image_urls, 'description':[size_tag_1, size_tag_2, size_tag_3, size_tag_4]}
        else:
            metadata["Size"] = {'description':[size_tag_1, size_tag_2, size_tag_3, size_tag_4]}
        
        if color_image_urls:
            metadata['Color'] = {'link': color_image_urls, 'description':color_tag}
        else:
            metadata['Color'] = {'description':color_tag}
        
        if behavior_image_urls:
            metadata['Behavior'] = {'link':behavior_image_urls, 'description':behavior_tag}
        else:
            metadata['Behavior'] = {'description':behavior_tag}
        
        if habitat_image_urls:
            metadata['Habitat'] = {'link': habitat_image_urls, 'description':habitat_tag}
        else:
            metadata['Habitat'] = {'description':habitat_tag}
        
        if text_tags:
            regional_tag = text_tags.find("p")
            
            if region_image_urls:
                metadata['Regional_Difference'] = {'link':region_image_urls, 'description':regional_tag}
            else:
                metadata['Regional_Difference'] = {'description':regional_tag}
        else: # in case there is no regional tag
            metadata = metadata
        
        #store metadata to flat file
        store_meta(species, metadata)
    except Exception as e:
        print('huh')
        print(str(e))      

# %%
id_scraper('American_Crow')

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
        if not os.path.exists(SPECIES_COMPARE_FOLDER+'/'+species):
            os.makedirs(SPECIES_COMPARE_FOLDER+'/'+species)
        # fetch the url and content
        if os.path.isfile(BIRD_PAGE_FOLDER+'/'+species+'/species-compare.html'):
            print(f'The page for {species} is already there!!!')
            with open(BIRD_PAGE_FOLDER+'/'+species+'/species_compare.html', 'rb') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        else:
            page = requests.get(URL, headers=HEADERS)
            time.sleep(1)
            with open(BIRD_PAGE_FOLDER+'/'+species+'/species-compare.html', 'wb+') as f:
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

                bird_type = child2.find('h5')
                if bird_type:
                    bird_type = bird_type.get_text()
                else:
                    bird_type = 'Unknown Type'
                
                bird_desc = child2.find('p').get_text()
                bird_desc = " ".join(bird_desc.split()) # remove duplicate spaces and tabs, newlines

                if bird_name not in similarbird_dict:
                    similarbird_dict[bird_name] = {}
                similarbird_dict[bird_name][bird_type] = {'link':img_links, 'description':bird_desc}

                # save images
                #make folders if they don't yet exist
                # if not os.path.exists(f"{SPECIES_COMPARE_FOLDER}/{species}/{bird_name}"):
                #     os.makedirs(f"{SPECIES_COMPARE_FOLDER}/{species}/{bird_name}")
                # for link in img_links:
                #     filename = link.split('/')[6]
                #     path = f"{SPECIES_COMPARE_FOLDER}/{species}/{bird_name}/{filename}"
                #     get_and_store_image(link, path)
                #     time.sleep(1)
        
        # save the descriptions
        json_object = json.dumps(similarbird_dict, indent=4)
        with open(f"{SPECIES_COMPARE_FOLDER}/{species}/similarbird_dict.json", 'w') as f:
            f.write(json_object)
        
            

    except Exception as e:
        print(str(e))   

#%%
species_compare_scraper('Tropical_Kingbird')

# %%
species = os.listdir('allaboutbirds_pages')
print(f'There are {len(species)} birds on the scraping list.')

for specy in species:
    id_scraper(specy)

#%%
for specy in species:
    species_compare_scraper(specy)

# %%
# check classes that cannot be parsed
ids = os.listdir('allaboutbirds_ids')
print(f'There are {len(ids)} birds on the scraping list.')
species_compares = os.listdir('allaboutbirds_species_compares')
print(f'There are {len(species_compares)} birds on the scraping list.')

num_id_failed = {'0': 0, '1': 0}
for id in ids:
    path = os.path.join('allaboutbirds_ids', id)
    if len(os.listdir(path)) == 0:
        num_id_failed['0'] += 1
    if len(os.listdir(path)) == 1:
        print(id)
        num_id_failed['1'] += 1
print(num_id_failed)

# %%
num_sc_failed = 0
for sc in species_compares:
    path = os.path.join('allaboutbirds_species_compares', sc)
    if len(os.listdir(path)) == 0:
        print(sc)
        num_sc_failed += 1
print(num_sc_failed)

# %%
