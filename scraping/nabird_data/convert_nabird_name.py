#%%
import os, time
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import shutil

# %%
HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
        # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        # "Accept-Language": "en-US,en;q=0.5",
        # "Accept-Encoding": "gzip, deflate",
        # "Connection": "keep-alive",
        # "Upgrade-Insecure-Requests": "1",
        # "Sec-Fetch-Dest": "document",
        # "Sec-Fetch-Mode": "navigate",
        # "Sec-Fetch-Site": "none",
        # "Sec-Fetch-User": "?1",
        # "Cache-Control": "max-age=0",
    }
# %%
class_file = '../merged_nabird_cub.txt' #'nabird_classes.txt'
f = open(f"{class_file}", 'r')
lines = f.readlines()

class_names = []
for index, line in enumerate(lines):
    if '\n' in line:
        line = line[:-1]
    name = line
    class_names.append(name)
    
    
print(f"Num orig names: {len(class_names)}")
print(f"Unique: Num orig names: {len(set(class_names))}")

# %%
class_names
# %%
def download_search_page(search_term, index, save_path):
# build the search URL
    url = f'https://www.allaboutbirds.org/news/search/?q={search_term}'
    headers = HEADERS

    # parse the HTML content of the response using BeautifulSoup
    try:
        if os.path.isfile(f'{save_path}/search_{search_term}.html'):
            print(f'The page for {search_term} is already there!!!')

            ## check if a scraped page is Forbidden or not 
            # with open(f'{save_path}/{index}_search_{search_term}.html', 'rb') as f:
            #     soup = BeautifulSoup(f.read(), 'html.parser')
            # is_forbidden = soup.find('title').get_text()
            # if is_forbidden == '403 Forbidden':
            #     print(f'The page for {search_term} is 403 Forbidden!!!')
            #     return 0
        else: 
            time.sleep(1)
            print(f'Scraping the page for {search_term}...')
            # send a GET request to the search URL
            page = requests.get(url, headers=headers, stream=True)
            soup = BeautifulSoup(page.content, 'html.parser')
            is_forbidden = soup.find('title').get_text()
            if is_forbidden == '403 Forbidden':
                print(f'The page for {search_term} is 403 Forbidden!!!')
                return 0
            with open(f'{save_path}/search_{search_term}.html', 'wb+') as f:
                f.write(page.content)
        return 1
    except Exception as e:
        print(str(e))
        return 0

# %%
SEARCH_FOLDER = './searching/'
# %%
#make folders if they don't yet exist
if not os.path.exists(SEARCH_FOLDER):
    os.makedirs(SEARCH_FOLDER)

num_fail = 0 # number of pages failing to be scraped
begin = time.time()
for i in range(len(class_names)):
    # if i < 478:
    #     continue
    name = class_names[i]
    
    status = download_search_page(name, i, SEARCH_FOLDER)
    if status == 0:
        num_fail += 1
    # else:
    #     shutil.copyfile(f'{SEARCH_FOLDER}/search_{name}.html', f'new_searching/search_{name}.html')
print(num_fail)
print(time.time() - begin)

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
site_files = os.listdir(SEARCH_FOLDER)
urls = {'Sites':[], 'Class names':[]}
for site_file in site_files:
    url = get_information_from_site(site_file)
    if url[-1] == '/':
        url = url[:-1]
    class_name = url.split('/')[-1]
    if url is None:
        print(f"URL of {class_name} is None")
        continue
    urls['Sites'].append(url)
    urls['Class names'].append(class_name)

print(f"Num sites: {len(site_files)}, and num sites having URL: {len(urls['Sites'])}")

# %%
urls
# %%
# write to dataframe
df = pd.DataFrame.from_dict(urls)
df

# %%
list_urls = df.Sites.values.tolist()
list_names = df['Class names'].values.tolist()

# check exact match between URLs and the class names
list_exact_match = []
num_exact_match = 0
for url, name in zip(list_urls, list_names):
    if url[-1] == '/':
        url = url[:-1]

    url = url.split('/')[-1]
    url = url.replace("_", " ")
    url = url.replace("-", " ")
    
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    
    if name == url:
        list_exact_match.append('exact')
        num_exact_match+=1
    else:
        list_exact_match.append('x')

print(num_exact_match)
# %%
df['Match'] = list_exact_match
df

# %%
df.to_csv('nabirds&cub_search_links.csv', index=False)  

# %%
