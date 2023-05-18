import numpy as np
from collections import OrderedDict
from PIL import Image

from torch.nn import functional as F
import torch
import torchvision.transforms as transforms
import clip

def _transform(n_px=224):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def compute_description_encodings(model, text_descriptions, device):
    description_encodings = OrderedDict()
    for v in text_descriptions:
        tokens = clip.tokenize(v).to(device)
        description_encodings[v] = F.normalize(model.encode_text(tokens))
    return description_encodings

def image_text_matching(model, preprocess, image: Image, text: str):
    device = next(model.parameters()).device

    texts = text.split('.')
    num_texts = len(texts)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # encode image and texts
    description_encodings = compute_description_encodings(model, texts, device)
    image_encodings = model.encode_image(image_tensor)
    image_encodings = F.normalize(image_encodings)

    image_description_similarity = [None]*num_texts
    for i, (v, text_encodings) in enumerate(description_encodings.items()):
        
        dot_product_matrix = image_encodings @ text_encodings.T        
        image_description_similarity[i] = round(dot_product_matrix.detach().cpu().item(), 3)
    
    return texts, image_description_similarity


if __name__ == '__main__':
    image = Image.open("/home/tin/reasoning/demo_image2text/static/demo1.jpg")
    text = 'a'

    model, preprocess = clip.load("ViT-B/32", device='cuda:7', jit=False)
    model.eval()
    model.requires_grad_(False)

    image_text_matching(model, preprocess, image, text)