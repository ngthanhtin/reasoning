# %% [markdown]
# 
# # Download the Unsplash dataset
# 
# This notebook can be used to download all images from the Unsplash dataset: https://github.com/unsplash/datasets. There are two versions Lite (25000 images) and Full (2M images). For the Full one you will need to apply for access (see [here](https://unsplash.com/data)). This will allow you to run CLIP on the whole dataset yourself. 
# 
# Put the .TSV files in the folder `unsplash-dataset/full` or `unsplash-dataset/lite` or adjust the path in the cell below. 

# %%
from pathlib import Path

dataset_version = "lite"  # either "lite" or "full"
unsplash_dataset_path = '/home/tin/datasets/unsplash/'

# %% [markdown]
# ## Load the dataset
# 
# The `photos.tsv000` contains metadata about the photos in the dataset, but not the photos themselves. We will use the URLs of the photos to download the actual images.

# %%
import pandas as pd

# Read the photos table
photos = pd.read_csv(unsplash_dataset_path  + "/photos.tsv000", sep='\t', header=0)

# Extract the IDs and the URLs of the photos
photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

# Print some statistics
print(f'Photos in the dataset: {len(photo_urls)}')

# %% [markdown]
# The file name of each photo corresponds to its unique ID from Unsplash. We will download the photos in a reduced resolution (640 pixels width), because they are downscaled by CLIP anyway.

# %%
import urllib.request
import os

# Path where the photos will be downloaded
photos_donwload_path = f"{unsplash_dataset_path}/photos"
if not os.path.exists(photos_donwload_path):
    os.makedirs(photos_donwload_path)
# Function that downloads a single photo
def download_photo(photo):
    # Get the ID of the photo
    photo_id = photo[0]

    # Get the URL of the photo (setting the width to 640 pixels)
    photo_url = photo[1] + "?w=640"

    # Path where the photo will be stored
    photo_path = f"{photos_donwload_path}/{photo_id}.jpg"

    # Only download a photo if it doesn't exist
    if not os.path.exists(photo_path):
        try:
            urllib.request.urlretrieve(photo_url, photo_path)
        except:
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}")
            pass

# %% [markdown]
# Now the actual download! The download can be parallelized very well, so we will use a thread pool. You may need to tune the `threads_count` parameter to achieve the optimzal performance based on your Internet connection. For me even 128 worked quite well.

# %%
from multiprocessing.pool import ThreadPool

# Create the thread pool
threads_count = 16
pool = ThreadPool(threads_count)

# Start the download
pool.map(download_photo, photo_urls)

# Display some statistics
display(f'Photos downloaded: {len(photos)}')



# %%
