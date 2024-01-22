import os
import fire
import json
import argparse

import clip
import torch
import torchmetrics
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import imagenet_templates, PROJECT_ROOT, indices_in_1k
from data_loader import DatasetWrapper

def eval(model: str = "ViT-B/32", 
         dataset: str = "imagenet", 
         batch_size: int = 128,
         device: str = "cuda:0",
         topk: list[int] = None,
         ):
    if topk is None:
        topk = [1, 5, 10, 20, 50]
    # load model
    model, preprocess = clip.load(model, device=device)
    model.eval()
    
    # load dataset and dataloader
    dataset_ = DatasetWrapper(dataset, preprocess)
    dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=True, num_workers=32)
    
    # load class names
    if dataset in {"imagenet", "imagenet-a", "imagenet-v2"}:
        json_name = "imagenet"
    else:
        json_name = dataset
    desc = json.load(open(f"{PROJECT_ROOT}/data/text/descriptors_{json_name}.json", "r"))
    class_names = list(desc.keys())
    if dataset == "imagenet-a":
        class_names = [class_name for idx, class_name in enumerate(class_names) if idx in indices_in_1k]
    
    # class name embeddings (average over 80 prompts)
    texts = {}
    for classname in class_names:
        texts[classname] = [template.format(classname) for template in imagenet_templates]
    text_embs = []
    idx2name = {}
    for idx, (classname, text) in tqdm(enumerate(texts.items()), desc="Computing text embeddings", total=len(texts)):
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(text)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.mean(dim=0)
            text_emb /= text_emb.norm()
        text_embs.append(text_emb)
        idx2name[idx] = classname
        
    text_embs = torch.stack(text_embs)
    
    accuracy_metrics = {k: torchmetrics.Accuracy(task="multiclass", num_classes=dataset_.n_classes, top_k=k) for k in topk}
    
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Evaluating CLIP", total=len(dataloader)):
        images, targets, paths, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            image_embs = model.encode_image(images)
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_embs @ text_embs.T).softmax(dim=-1).cpu().float()
        for k in topk:
            accuracy_metrics[k].update(logits, targets)
    
    for k in topk:
        print(f"Top-{k} accuracy: {accuracy_metrics[k].compute()}")
        

if __name__ == "__main__":
    fire.Fire(dict(eval=eval))