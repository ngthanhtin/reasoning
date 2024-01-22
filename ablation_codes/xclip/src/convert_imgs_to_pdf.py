import glob
import img2pdf
import random
from utils import PROJECT_ROOT


# datasets = ["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "imagenet-c", "imagenet-c", "places365", "cub"]
# folders = ["02_15_2023-02:28:06_viz", "02_15_2023-02:27:50_viz", "02_15_2023-02:27:51_viz",
#            "02_20_2023-21:45:09_viz", "02_20_2023-21:45:33_viz", "02_20_2023-21:45:20_viz", # gaussian_noise / glass_blur / defocus_blur
#            "02_15_2023-02:28:04_viz", "02_15_2023-02:25:43_viz"]
# sub_folders = ["both_correct", "both_wrong", "sachit_correct", "xclipv2_correct"]
#
# distortions = {"02_20_2023-21:45:09_viz": "gaussian_noise",
#                "02_20_2023-21:45:33_viz": "glass_blur",
#                "02_20_2023-21:45:20_viz": "defocus_blur"}
#
# # Convert all files matching a glob
# for dataset, folder in zip(datasets, folders):
#     if folder == "" or dataset not in ["imagenet-c"]:
#         continue
#
#     for sub_folder in sub_folders:
#         with open(f"{PROJECT_ROOT}/results/{dataset}_{distortions[folder]}_{sub_folder}_subset.pdf", "wb") as f:
#             img_files = glob.glob(f"{PROJECT_ROOT}/results/{dataset}/{folder}/explanations/{sub_folder}/*.jpg")
#             img_files = random.sample(img_files, k=int(len(img_files) * 0.1))
#             f.write(img2pdf.convert(img_files))


datasets = ["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "imagenet-c", "imagenet-c", "places365", "cub"]
folders = ["02_24_2023-16:36:18_owl_vit", "02_24_2023-19:16:40_owl_vit", "02_24_2023-16:36:24_owl_vit",
           "02_24_2023-19:16:50_owl_vit", "02_24_2023-19:16:42_owl_vit", "02_24_2023-19:16:46_owl_vit",     # gaussian_noise / defocus_blur / glass_blur
           "02_24_2023-16:36:28_owl_vit", "02_24_2023-16:36:34_owl_vit"]

distortions = {"02_24_2023-19:16:50_owl_vit": "gaussian_noise",
               "02_24_2023-19:16:42_owl_vit": "defocus_blur",
               "02_24_2023-19:16:46_owl_vit": "glass_blur"}

# Convert all files matching a glob
for dataset, folder in zip(datasets, folders):
    if folder == "":
        continue

    output_file = f"{PROJECT_ROOT}/results/{dataset}_owlvit_subset.pdf"
    if dataset == "imagenet-c":
        output_file = f"{PROJECT_ROOT}/results/{dataset}_{distortions[folder]}_owlvit_subset.pdf"

    with open(output_file, "wb") as f:
        img_files = glob.glob(f"{PROJECT_ROOT}/results/{dataset}/{folder}/*.jpg")
        img_files = random.sample(img_files, k=int(len(img_files) * 0.1))
        f.write(img2pdf.convert(img_files))

