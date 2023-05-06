#%%
import os
import requests
from bs4 import BeautifulSoup
import re

f = open("nabird_classes.txt", 'r')
lines = f.readlines()

orig_name = []
new_name = []
for line in lines:
    if '\n' in line:
        line = line[:-1]
    id, name = line.split(' ', 1)
    orig_name.append(name)
    if ',' in name or ' and' in name:
        continue
    if '(' in name:
        cut_idx = name.index('(')
        name = name[:cut_idx-1]
    
    new_name.append(name)
print(f"Num orig names: {len(orig_name)}, Num new name: {len(new_name)}")
print(f"Unique: Num orig names: {len(set(orig_name))}, Num new name: {len(set(new_name))}")
# %%
def download_search_page(search_term, save_path):
# build the search URL
    url = f'https://www.allaboutbirds.org/news/search/?q={search_term}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}

    # send a GET request to the search URL
    page = requests.get(url, headers=headers, stream=True)

    # parse the HTML content of the response using BeautifulSoup
    try:
        if os.path.isfile(f'{save_path}/search_{search_term}.html'):
            print(f'The page for {search_term} is already there!!!')
        else:   
            with open(f'{save_path}/search_{search_term}.html', 'wb+') as f:
                f.write(page.content)
        return 1
    except Exception as e:
        print(str(e))
        return 0

# %%

SEARCH_FOLDER = './searching/'
#make folders if they don't yet exist
if not os.path.exists(SEARCH_FOLDER):
    os.makedirs(SEARCH_FOLDER)

num_fail = 0 # number of pages failing to be scraped
for i in range(len(orig_name)):
    name = orig_name[i]
    if "/" in name:
        name = name.replace("/", " and " )
    
    status = download_search_page(name, SEARCH_FOLDER)
    if status == 0:
        num_fail += 1
print(num_fail)
# %%
def get_information_from_site(site_name):
    with open(f'{SEARCH_FOLDER}/{site_name}', 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # find all the search result items on the page
    search_results = soup.find_all('div', id='species-search-results')

    # Get URLs of the search result items
    url = None
    for result in search_results:
        first_item = result.find('a', class_='article-item-link')
        url = first_item.get('href')

    return url
# %%
site_names = os.listdir(SEARCH_FOLDER)
urls = []
for site_name in site_names:
    url = get_information_from_site(site_name)
    if url is None:
        continue
    urls.append(url)

print(f"Num sites: {len(site_names)}, and num sites having URL: {len(urls)}")

# %%
with open("nabirds_search_links.txt", 'w') as f:
    for url in urls:
        f.write(url)
        f.write('\n')
        

# %%
