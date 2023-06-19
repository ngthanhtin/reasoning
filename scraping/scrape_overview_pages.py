# %%
import os, json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
# %%
# scratch overview pages of allaboutbirds
HEADERS = {'accept': '"text/html', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}
CUB_PAGE_FOLDER = 'cub_pages/'
ALLABOUTBIRDS_FOLDER = 'allaboutbirds_pages/'
ALLABOUTBIRDS_OVERVIEW_FOLDER = 'allaboutbirds_overview_pages/'

# %%
def get_scientific_name_from_page(path):
    bird_name = path.split('/')[-1].split('.')[0]
    with open(path, 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        info_tag = soup.find('div', {"class":"species-info"})
        scientific_name = info_tag.find('em').get_text()

    return scientific_name, bird_name

# scrape overview pages
def overview_page_scraper(species, url, save_folder):

    is_failed = False
    
    # fetch the overview url
    if os.path.isfile(save_folder+'/'+species+'.html'):
        print(f'The overview page for {species} is already there!!!')
    else:
        print(f'Scraping overview page {url}')
        page = requests.get(url, headers=HEADERS)
        if page.status_code != 200:
            print(f"Download error {page.status_code} {url}")
            is_failed = True
        else:
            with open(save_folder+'/'+species+'.html', 'wb+') as f:
                f.write(page.content)
            time.sleep(1)

    return is_failed

# %%
def scrape_allaboutbirds_overview_pages():
    bird_names = os.listdir(ALLABOUTBIRDS_FOLDER)
    num_fails = 0
    for bird_name in bird_names:
        url = f'https://www.allaboutbirds.org/guide/{bird_name}'
        try:
            overview_page_scraper(bird_name, url, save_folder=ALLABOUTBIRDS_OVERVIEW_FOLDER)
        except:
            num_fails+=1
    print(f"The number of failed scraped pages: {num_fails}")

def get_scientific_name_allaboutbirds():
    # parse html to get scientific names
    url_paths = os.listdir(ALLABOUTBIRDS_OVERVIEW_FOLDER)
    url_paths = [os.path.join(ALLABOUTBIRDS_OVERVIEW_FOLDER, url) for url in url_paths]

    birdname_2_sciname = {}
    for path in url_paths:
        scientific_name, bird_name = get_scientific_name_from_page(path)
        birdname_2_sciname[bird_name] = scientific_name
    
    # save to json
    json_object = json.dumps(birdname_2_sciname, indent=4)
    with open("birdname_2_sciname_allaboutbirds.json", "w") as outfile:
        outfile.write(json_object)


# %%
scrape_allaboutbirds_overview_pages()

# %%
get_scientific_name_allaboutbirds()

# %%
def scrape_and_get_scientific_name_cub_overview_pages():
    # Find CUB URL
    final_nabird_cub_df = pd.read_csv("./final_nabirds_cub_search_links.csv")
    final_nabird_cub_df.head(5)

    auto_sites = final_nabird_cub_df.Sites.values.tolist()
    matchings = final_nabird_cub_df.Match.values.tolist()
    manual_sites = final_nabird_cub_df['Check URL manually'].values.tolist()
    changed_class_names = final_nabird_cub_df['Class names'].values.tolist()

    right_sites = []
    for auto_url, matching, manual_url in zip(auto_sites, matchings, manual_sites):
        if matching == 'exact':
            right_sites.append(auto_url)
        else:
            right_sites.append(manual_url)
    print(right_sites[:5], changed_class_names[:5])

    # CUB classes
    cub_classes_path = './cub_data/cub_classes.txt'
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

    print(cub_table)

    cub_classes = cub_table.name.values.tolist()
    print(cub_classes)
    # %%
    fix_cub_classes = [cls.replace("_", " ") for cls in cub_classes]
    fix_cub_classes = [cls.replace("-", " ") for cls in fix_cub_classes]
    print(fix_cub_classes)

    num_success = 0
    cub_sites = []
    for cls in fix_cub_classes:
        if cls in changed_class_names:
            index = changed_class_names.index(cls)
            cub_sites.append(right_sites[index])
            num_success+=1

    print(num_success - len(fix_cub_classes), cub_sites[:5], fix_cub_classes[:5], cub_classes[:5])

    # Check if any classes do not exists on AllaboutBirds
    num = 0
    for i,site in enumerate(cub_sites):
        if site == 'x':
            print(cub_classes[i])
            num+=1
    print(num)

    cub_dict = {'Site': cub_sites, 'Original Name': cub_classes, 'Fixed Name': fix_cub_classes}

    cub_df = pd.DataFrame.from_dict(cub_dict)

    for i in range(len(cub_df)):
        url, orig_name, fixed_name = cub_df.iloc[i].tolist()
        if url != 'x':
            overview_page_scraper(fixed_name, url, save_folder=CUB_PAGE_FOLDER)

    # parse html to get scientific names
    url_paths = os.listdir(CUB_PAGE_FOLDER)
    url_paths = [os.path.join(CUB_PAGE_FOLDER, url) for url in url_paths]

    birdname_2_sciname = {}
    for path in url_paths:
        scientific_name, bird_name = get_scientific_name_from_page(path)
        birdname_2_sciname[bird_name] = scientific_name

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
