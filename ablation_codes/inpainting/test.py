# %%
import os

nabirds_path = '/home/tin/datasets/nabirds/images/'
inpaint_path = 'nabirds_inpaint_kp_full/'

orig_folders = os.listdir(nabirds_path)
inpaint_folders = os.listdir(inpaint_path)

not_inpaint_folders = []

for orig_f in orig_folders:
    if orig_f not in inpaint_folders:
        not_inpaint_folders.append(orig_f)

len(orig_folders), len(not_inpaint_folders)
# %%
num_orig = 0
for f in os.listdir(nabirds_path):
    num_orig += len(os.listdir(f"{nabirds_path}/{f}"))

num_inpaint = 0
for f in os.listdir(inpaint_path):
    num_inpaint += len(os.listdir(f"{inpaint_path}/{f}"))

num_orig, num_inpaint
# %%
# check error files of Nabirds inpaint
import os
orig_nabirds_path = '/home/tin/datasets/nabirds/train/'
inpaint_nabirds_path = '/home/tin/datasets/nabirds/train_inpaint/'

num_error = 0
label_folders = os.listdir(orig_nabirds_path)
for folder in label_folders:
    orig_folder_path = os.path.join(orig_nabirds_path, folder)
    inpaint_folder_path = os.path.join(inpaint_nabirds_path, folder)

    orig_imgs = os.listdir(orig_folder_path)
    inpaint_imgs = os.listdir(inpaint_folder_path)

    for orig in orig_imgs:
        if orig not in inpaint_imgs:
            num_error += 1
            print(f"{orig_folder_path}/{orig}")


num_error

# %%
