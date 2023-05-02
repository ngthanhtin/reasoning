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

# how many images to retrieve per species
MAXIMAGES = 10000

# the number of images returned per page (needed for pagination)
IMAGESPERPAGE = 24

species = [{
    'name': 'pimpelmees',
    'id': '161'
}, {
    'name': 'koolmees',
    'id': '140'
}, {
    'name': 'staartmees',
    'id': '181'
}, {
    'name': 'kuifmees',
    'id': '145'
}, {
    'name': 'vink',
    'id': '193'
}, {
    'name': 'merel',
    'id': '150'
}, {
    'name': 'spreeuw',
    'id': '180'
}, {
    'name': 'ringmus',
    'id': '166'
}, {
    'name': 'huismus',
    'id': '122'
}, {
    'name': 'geelgors',
    'id': '55'
}, {
    'name': 'groenling',
    'id': '261771'
}, {
    'name': 'heggenmus',
    'id': '118'
}, {
    'name': 'boomkruiper',
    'id': '71'
}, {
    'name': 'boomklever',
    'id': '70'
}, {
    'name': 'roodborstje',
    'id': '168'
}, {
    'name': 'grotebontespecht',
    'id': '109'
}, {
    'name': 'ekster',
    'id': '87'
}, {
    'name': 'putter',
    'id': '162'
},{
    'name': 'winterkoning',
    'id': '199'
},{
    'name': 'zanglijster',
    'id': '204'
},{
    'name': 'houtduif',
    'id': '120'
},{
    'name': 'turksetortel',
    'id': '191'
},{
    'name': 'holenduif',
    'id': '119'
},{
    'name': 'wittekwikstaart',
    'id': '202'
},{
    'name': 'groenespecht',
    'id': '40'
},{
    'name': 'vlaamsegaai',
    'id': '92'
},{
    'name': 'keep',
    'id': '127'
},{
    'name': 'koperwiek',
    'id': '141'
},{
    'name': 'kramsvogel',
    'id': '143'
},{
    'name': 'wielewaal',
    'id': '350'
},{
    'name': 'kauw',
    'id': '126'
},{
    'name': 'zwartekraai',
    'id': '208'
},{
    'name': 'kokmeeuw',
    'id': '138'
}]

print(f'There are {len(species)} birds on the scraping list.')

#%%
def get_and_store_image(url: str, path: str):
    '''Obtain an image fm the world wide web and store it in the path specified'''
    response = requests.get(url, stream=True)
    with open(path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response  
    
def center_image(img):
    '''Convenience function to return a centered image'''
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized
# %%
def show_random_img_from_folder(folder: str):
    '''Show a random image from the given folder'''
    
    filelist = [f for f in listdir(folder) if isfile(join(folder, f))]
    imglocation = random.choice(filelist)
    
    img=mpimg.imread(folder+'/'+imglocation)
    imgplot = plt.imshow(img)
    plt.show()
    
    print(imglocation)
# %%
def convert_and_store_img(inputpath: str, outputpath: str):
    '''Converts an image and resizes it, stores it to disk (ssd in this case)'''
    
    # try-catch is necessary here because: 
    # a) images might be really small for some reason
    # b) images might be served with a 0-dimension from waarneming.nl (e.g. 512x0) possibly corrupted
    #    during upload / download
    # bit of an antipattern for sure.
    try:
        img = cv2.imread(inputpath)
        
        #calculate tile size
        if(img.shape[0] > img.shape[1]):
            tile_size = (int(img.shape[1]*256/img.shape[0]),256)
        else:
            tile_size = (256, int(img.shape[0]*256/img.shape[1]))

        #centering + actual resizing
        img = center_image(cv2.resize(img, dsize=tile_size))

        #output should be 224*224px for a quick vggnet16
        img = img[16:240, 16:240]

        cv2.imwrite(outputpath, img)
        
    except:
        print(inputpath)
        pass
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
    photolinks = []
    # text
    results = True
    
    while results:
        try:
            #pause for one second out of courtesy
            time.sleep(1)

            # construct the url
            URL = f'https://www.allaboutbirds.org/guide/{species}/id'

            # fetch the url and content
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, 'html.parser')
            photo_tags=soup.findAll('img', {"alt":"Dark-eyed Junco"})
            text_tags=soup.find('article', {"aria-label":"Size & Shape"})
            # get the text size & shape
            children = text_tags.find("p")
            children = text_tags.find("div").find("p")
            children = text_tags.find("div").find("span")

            if len(photo_tags) == 0 or len(text_tags) == 0:
                results = False
                break

            photolinks += photo_tags
        except Exception as e:
            print(str(e))
            
        results = False
        
    #Show a count of how many we've found
    print(f'Found {photolinks} photos for {species}.')

    #make folders if they don't yet exist
    if not os.path.exists(RAWFOLDER+'/'+species):
        os.makedirs(RAWFOLDER+'/'+species)
        
    # get the photoids we already have scraped from - the links change
    photoids = get_photoid_list(species)
    for link in photolinks:
        image_url = link['data-interchange'].split(',')[0][1:]
        
        #obtain filename from url
        filename = image_url.split('/')[6]
        #check if we have encountered this photo before- will be substantially slower with large n
        if filename.split('.')[0] not in photoids: 

            path = RAWFOLDER+'/'+species+'/'+filename
            get_and_store_image(image_url, path)

            #pause for one second out of courtesy
            time.sleep(1)
    
    #store metadata to flat file
    # store_meta(species, metadata)
# %%
id_scraper('Dark-eyed_Junco')
# %%
def store_meta(species: str, meta: list):
    '''Store metadata to file for later use/verification'''
    
    with open(f'{species}.txt', 'a+', encoding="utf-8") as f:
        for item in meta:
            f.write("%s\n" % item)
# %%
def get_metadata(photoid=24691898) -> dict:
    '''Given a photo-id, return metadata in a dict'''
    
    url = 'https://waarneming.nl/photos/'+str(photoid)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    tags=soup.find('table',{"class":"table app-content-section photo-detail"})
    meta =  {}

    if tags:
        # find all table rows
        rows = tags.find_all('tr')
        values = []
        keys = []
        # get the table content and return as two lists
        for row in rows:
            # actual content is listed in the <td>, while <th> holds the titles. 
            descriptions = row.find_all('th')
            cols = row.find_all('td')
            for idx, ele in enumerate(descriptions):
                keys.append(ele.text.strip())
                values.append(cols[idx].text.strip())

        #create a dict out of the data we fetched
        meta = dict(zip(keys, values))  

    return meta
# %%
#check if the metadata call returns the right stuff
# get_metadata()

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
