import os
import clip
import json

from PIL import Image
from torch.nn import functional as F
import torchvision.transforms as transforms

from configs import *


imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

''' CLIP TRANSFORM
0 = {Resize} Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
1 = {CenterCrop} CenterCrop(size=(224, 224))
2 = {function} <function _convert_image_to_rgb at 0x7fd16ab12830>
3 = {ToTensor} ToTensor()
4 = {Normalize} Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
'''

def img_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def compute_description_encodings(model, descriptions, templates=None):
    description_encodings = {}

    for k, v in descriptions.items():
        if templates is not None:
            description_encodings[k] = compute_templated_encodings(v, model, templates)
        else:
            tokens = clip.tokenize(v).to(device)
            description_encodings[k] = F.normalize(model.encode_text(tokens)).detach().cpu()

    return description_encodings


def compute_label_encodings(model, class_names, templates=None, device='cuda', tokenizer=None):
    model.to(device)
    if templates is not None:
        labels = [wordify(class_name) for class_name in class_names]
        return compute_templated_encodings(labels, model, templates=templates, tokenizer=tokenizer)
    else:
        label_encodings = F.normalize(model.encode_text(tokenizer([wordify(l) for l in class_names]).to(device)))

    return label_encodings


def compute_templated_encodings(texts, model, templates=None, tokenizer=None):
    all_encodings = []

    for text in texts:
        templated_text = [template.format(text) for template in templates]
        tokens = tokenizer(templated_text, truncate=True).to(device)                    # tokenize
        templated_text_encodings = model.encode_text(tokens).detach().cpu()                 # embed with text encoder
        templated_text_encodings /= templated_text_encodings.norm(dim=-1, keepdim=True)
        mean_text_embedding = templated_text_encodings.mean(dim=0)
        mean_text_embedding /= mean_text_embedding.norm()
        all_encodings.append(mean_text_embedding)

    all_encodings = torch.stack(all_encodings, dim=0)

    return all_encodings


def load_descriptions(dataset_name, classes_to_load=None, prompt_type=None, desc_type="sachit"):
    templated_descriptions, descriptions_mappings = {}, {}

    # ImageNet and ImageNet-v2 share the same list of descriptions
    dataset_to_load = "imagenet" if dataset_name in ["imagenet-v2", "imagenet-a", "imagenet-c"] else dataset_name
    # with open(f"{PROJECT_ROOT}/data/text/{desc_type}/descriptors_{dataset_to_load}.json") as input_file:
    if dataset_name == 'inaturalist2021':
        with open(f"{PROJECT_ROOT}/data/text/{desc_type}/425_additional_{desc_type}_descriptors_inaturalist.json") as input_file:
            descriptions = json.load(input_file)
    else:
        with open(f"{PROJECT_ROOT}/data/text/{desc_type}/thisbird_additional_{desc_type}_descriptors_{dataset_to_load}.json") as input_file:
            descriptions = json.load(input_file)

    if classes_to_load is not None:
        descriptions = {c: descriptions[c] for c in classes_to_load}
    elif dataset_name == "imagenet-a":
        descriptions = {c: descriptions[c] for idx, c in enumerate(descriptions.keys()) if idx in indices_in_1k}

    if prompt_type is not None:
        for i, (class_name, class_descriptors) in enumerate(descriptions.items()):
            if len(class_descriptors) == 0:
                class_descriptors = ['']

            class_name = wordify(class_name)

            # Sachit's prompt
            if prompt_type == 0:
                templated_descriptors = class_descriptors
            elif prompt_type == 1:
                templated_descriptors = [f"{make_descriptor_sentence(class_name, descriptor)}" for descriptor in class_descriptors]
            elif prompt_type == 2:
                templated_descriptors = [f"{descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 3:
                templated_descriptors = [f"a photo of {descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 4:
                templated_descriptors = [f"a cropped photo of {descriptor} of {class_name}" for descriptor in class_descriptors]
            elif prompt_type == 5:
                templated_descriptors = [f"a photo of a {make_descriptor_sentence(class_name, descriptor)}" for descriptor in class_descriptors]
            elif prompt_type == 6:
                templated_descriptors = [f"a cropped photo of a {make_descriptor_sentence(class_name, descriptor)}" for descriptor in class_descriptors]
            else:
                templated_descriptors = class_descriptors

            templated_descriptions[class_name] = templated_descriptors
            descriptions_mappings[class_name] = {templated_descriptor: descriptor for descriptor, templated_descriptor in zip(class_descriptors, templated_descriptors)}

            # Print an example for checking
            if i == 0:
                print(f"\nExample description for prompt type {prompt_type}: \"{templated_descriptions[class_name][0]}\"\n")

    return templated_descriptions, descriptions_mappings


def wordify(string):
    word = string.replace('_', ' ')
    return word


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean', dim=1):
    if aggregation_method == 'max':
        return similarity_matrix_chunk.max(dim=dim)[0]
    elif aggregation_method == 'sum':
        return similarity_matrix_chunk.sum(dim=dim)
    elif aggregation_method == 'mean':
        return similarity_matrix_chunk.mean(dim=dim)
    else:
        raise ValueError("Unknown aggregate_similarity")


def make_descriptor_sentence(class_name, descriptor):
    if descriptor.startswith(('a', 'an', 'used')):
        return f"{class_name}, which is {descriptor}"
    elif descriptor.startswith(('has', 'often', 'typically', 'may', 'can')):
        return f"{class_name}, which {descriptor}"
    else:
        return f"{class_name}, which has {descriptor}"


def check_device_availability(devices: list[int or str] or int or str):
    # check if the devices are in format 'cuda:X' or 'X' or X, where X is an integer
    def _check_device_format(device_name: str or int) -> int:
        if isinstance(device_name, int):
            return device_name
        elif isinstance(device_name, str):
            if device_name.startswith("cuda:"):
                return int(device_name.split(":")[1])
            else:
                return int(device_name)      
    
    # check devices if devices is a list
    if not isinstance(devices, list):
        devices = [devices]
    devices = [_check_device_format(device) for device in devices]
    # check if the devices are available
    
    available_list = list(range(torch.cuda.device_count()))
    # check if the device is available
    for device in devices:
        if device not in available_list:
            raise ValueError(f"Device {device} is not available. Available devices are {available_list}")


# check if precomputed embeddings are available in either PROJECT ROOT (root_path) or
# in the the /home/lab/xclip/segment_embeddings folder
def check_availability_of_precomputed_embs(emb_type: str, dataset_name: str, root_path: str) -> str or None:
    lab_folder = "/home/lab/xclip/segment_embeddings"
    if os.path.exists(os.path.join(root_path, dataset_name)):
        usr_emb_list = os.listdir(os.path.join(root_path, dataset_name))
    else:
        usr_emb_list = []
        
    if os.path.exists(os.path.join(lab_folder, dataset_name)):
        shared_emb_list = os.listdir(os.path.join(lab_folder, dataset_name))
    else:
        shared_emb_list = []
    # return path to the embedding folder
    if emb_type in shared_emb_list:
        print(f"Found {emb_type} embeddings in {lab_folder}")
        return os.path.join(lab_folder, dataset_name, emb_type)
    elif emb_type in usr_emb_list:
        print(f"Found {emb_type} embeddings in {root_path}")
        return os.path.join(root_path, dataset_name, emb_type)
    else:
        raise FileNotFoundError(f"Could not find {emb_type} embeddings in {root_path} or {lab_folder}, available embeddings are {usr_emb_list + shared_emb_list}")


def check_file_or_list(file_or_list: str or list[str]) -> list:
    # if the input is a list of length 1 or str, check if the file exists
    # if the file exist, load the file and return the list
    # otherwise, return the list
    if isinstance(file_or_list, list):
        if len(file_or_list) != 1:
            return file_or_list
        if len(file_or_list) == 1 and not os.path.exists(file_or_list[0]):
            print(f"Cannot find file {file_or_list}, return it as a list of class name.")
            return file_or_list
        with open(file_or_list[0], 'r') as f:
            return [line.strip() for line in f.readlines()]
    elif isinstance(file_or_list, str):
        if os.path.exists(file_or_list):
            with open(file_or_list, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Could not find {file_or_list}")


def intersect_list(target_list: list[str], full_list: list[str]) -> list[str]:
    # return the intersection of two lists, print out all element in the target list but not in the full list
    if not target_list:
        return full_list    # if the target list is empty, return the full list

    inter_list = []

    for item in target_list:
        if item in full_list:
            inter_list.append(item)
        else:
            print(f"{item} not found in descritors")

    return inter_list


def gaussian_blur(img_tensor: torch.Tensor, boxes: torch.Tensor = None, kernel_size: int = 9, sigma: float = 5.0) -> torch.Tensor:
    # img_tensor (C, H, W)
    # boxes (N, 4) (x1, y1, x2, y2)
    # kernel_size (int), sigma (float) for GaussianBlur
    # return a blured image (C, H, W) or a batch of blured images (N, C, H, W)
    blur = transforms.GaussianBlur(kernel_size, sigma)
    blur_img = blur(img_tensor)

    if boxes is None:
        return blur_img

    partial_blured_img = []

    for box in boxes:
        x1, y1, x2, y2 = box
        # replace the box with original image in blur_img
        background_blur = blur_img.clone()
        background_blur[:, y1:y2, x1:x2] = img_tensor[:, y1:y2, x1:x2]
        partial_blured_img.append(background_blur)

    return torch.stack(partial_blured_img, dim=0)

