# %%
import os
import pandas as pd

# %%
path = './scraping/allaboutbirds_data/'
id_path = path + "id_data/"
sc_path = path + "species_compare_data/"

bird_names = os.listdir(id_path)

num_complete = 0
for bird_name in bird_names:
    full_path = id_path + bird_name
    if len(os.listdir(full_path)) == 2:
        num_complete += 1
print(num_complete)

# %%
# check the number of images in allaboutbirds_data
bird_names = os.listdir(sc_path)

num_images = 0
for bird_name in bird_names:
    num_images += len(os.listdir(os.path.join(sc_path, bird_name)))
    num_images += len(os.listdir(os.path.join(id_path, bird_name + '/bird_type_data/')))
    num_images += len(os.listdir(os.path.join(id_path, bird_name + '/metadata_data/')))

print("The number of scraped images: ", num_images)

# %%
# Find CUB URL
final_nabird_cub_df = pd.read_csv("./scraping/final_nabirds_cub_search_links.csv")
final_nabird_cub_df.head(5)
# %%
auto_sites = final_nabird_cub_df.Sites.values.tolist()
matchings = final_nabird_cub_df.Match.values.tolist()
manual_sites = final_nabird_cub_df['Check URL manually'].values.tolist()
changed_class_names = final_nabird_cub_df['Class names'].values.tolist()
# %%
right_sites = []
for auto_url, matching, manual_url in zip(auto_sites, matchings, manual_sites):
    if matching == 'exact':
        right_sites.append(auto_url)
    else:
        right_sites.append(manual_url)
right_sites[:5], changed_class_names[:5]
# %%
# CUB classes
cub_classes_path = 'scraping/cub_data/cub_classes.txt'
# read cub class name
def split_id_name(s):
    try:
        return s.split('.')[1]
    except Exception as e:
        return None
    
cub_table = pd.read_table(f'{cub_classes_path}', sep=' ',
                            header=None)
cub_table.columns = ['id', 'name']
cub_table['name'] = cub_table['name'].apply(split_id_name)

cub_table

# %%
cub_classes = cub_table.name.values.tolist()
cub_classes
# %%
fix_cub_classes = [cls.replace("_", " ") for cls in cub_classes]
fix_cub_classes = [cls.replace("-", " ") for cls in fix_cub_classes]
fix_cub_classes
# %%
num_success = 0
cub_sites = []
for cls in fix_cub_classes:
    if cls in changed_class_names:
        index = changed_class_names.index(cls)
        cub_sites.append(right_sites[index])
        num_success+=1

num_success - len(fix_cub_classes), cub_sites[:5], fix_cub_classes[:5], cub_classes[:5]
# %%
# Check if any classes do not exists on AllaboutBirds
num = 0
for i,site in enumerate(cub_sites):
    if site == 'x':
        print(cub_classes[i])
        num+=1
num

# %%
cub_dict = {'Site': cub_sites, 'Original Name': cub_classes, 'Fixed Name': fix_cub_classes}
cub_dict
#%%
cub_df = pd.DataFrame.from_dict(cub_dict)
cub_df.head(5)


# %%
# scratch scientific name
import requests
from bs4 import BeautifulSoup
import time

HEADERS = {'accept': '"text/html', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}
CUB_PAGE_FOLDER = 'cub_pages/'
# scrape id and species compare pages
def page_scraper(species, url):

    is_failed = False
    
    # fetch the ID url
    if os.path.isfile(CUB_PAGE_FOLDER+'/'+species+'.html'):
        print(f'The CUB page for {species} is already there!!!')
    else:
        print(f'Scraping CUB page {url}')
        page = requests.get(url, headers=HEADERS)
        if page.status_code != 200:
            print(f"Download error {page.status_code} {url}")
            is_failed = True
        else:
            with open(CUB_PAGE_FOLDER+'/'+species+'.html', 'wb+') as f:
                f.write(page.content)
            time.sleep(1)

    return is_failed

# %%
for i in range(len(cub_df)):
    url, orig_name, fixed_name = cub_df.iloc[i].tolist()
    if url != 'x':
        page_scraper(fixed_name, url)


# %%
# parse html to get scientific names
url_paths = os.listdir(CUB_PAGE_FOLDER)
url_paths = [os.path.join(CUB_PAGE_FOLDER, url) for url in url_paths]

birdname_2_sciname = {}
for path in url_paths:
    bird_name = path.split('/')[-1].split('.')[0]
    with open(path, 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        info_tag = soup.find('div', {"class":"species-info"})
        scientific_name = info_tag.find('em').get_text()
        birdname_2_sciname[bird_name] = scientific_name

# %%
scientific_names = []
for i in range(len(cub_df)):
    url, orig_name, fixed_name = cub_df.iloc[i].tolist()
    if url == 'x':
        scientific_names.append('x')
    else:
        scientific_names.append(birdname_2_sciname[fixed_name])

print(len(scientific_names))
cub_df['Scientific Name'] = scientific_names
cub_df.to_csv("cub_df.csv", drop_index=True)
# %%
