import os
from typing import Callable

import clip
from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
from torch.nn import functional as F
from configs import *
from utils import compute_label_encodings, gaussian_blur
import matplotlib.pyplot as plt

import cv2

class BoxWrapper(VisionDataset):
    allowed_keywords = ["idx2name", "name2idx", "device", "blur_background", "blur_image", "blur_kernel", "dataset", "indexes_to_keep", "add_allaboutbirds_descs"]

    def __init__(self, boxes_dir, clip_model, preprocess, templated_descriptions, owlvit_threshold, clip_topk: int, clip_tokenizer: Callable = None, **kwargs):
        self.boxes_dir = boxes_dir
        self.clip_topk = clip_topk
        self.tokenizer = clip_tokenizer if clip_tokenizer is not None else clip.tokenize
        self.image_ids = [file_name.replace(".pth", "") for file_name in sorted(os.listdir(boxes_dir))]

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keywords)
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.pil2tensor = transforms.ToTensor()

        self.templated_descriptions = templated_descriptions
        self.owlvit_threshold = owlvit_threshold
        self.max_num_descriptors = max([len(descriptors) for descriptors in self.templated_descriptions.values()])
        class_list = list(self.templated_descriptions.keys())
        
        if not hasattr(self, 'device'):
            # Compatible with configs.py, remove this in future versions (Use device argument from argparser)
            self.device = device 

        print('Computing CLIP text embeddings...')
        self.label_embeds = compute_label_encodings(clip_model, class_list, device=self.device, tokenizer=self.tokenizer).detach()
        print('Done.')

    def __getitem__(self, idx):
        with torch.no_grad():
            owlvit_results = torch.load(f"{self.boxes_dir}/{self.image_ids[idx]}.pth")
            
            boxes_pred_dict = owlvit_results["boxes_info"]
            gt_class_name = owlvit_results["class_name"]
            if hasattr(self, 'dataset'):
                if self.dataset == 'cub':
                    owlvit_results["image_path"] = "/home/tin/datasets/cub/CUB/images/" + owlvit_results["image_path"].split("/")[-2] + "/" + owlvit_results["image_path"].split("/")[-1]
                elif self.dataset == 'nabirds':
                    # owlvit_results["image_path"] = owlvit_results["image_path"].replace('lab', 'tin')
                    owlvit_results["image_path"] = "/home/tin/datasets/nabirds/images/" + owlvit_results["image_path"].split("/")[-2] + "/" + owlvit_results["image_path"].split("/")[-1]
                elif self.dataset == 'inaturalist2021':
                    owlvit_results["image_path"] = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/" + owlvit_results["image_path"].split("/")[-2] + "/" + owlvit_results["image_path"].split("/")[-1]
                    
            image = Image.open(owlvit_results["image_path"]).convert('RGB')
            w,h = image.size
            img_tensor = self.pil2tensor(image)
            
            blurred_image_embed = None
            if hasattr(self, "blur_image") and self.blur_image:
                kernel_size = self.blur_kernel if hasattr(self, "blur_kernel") else 7
                blurred_img_tensor = gaussian_blur(img_tensor, kernel_size=kernel_size)
                blurred_image_embed = self.clip_model.encode_image(self.preprocess(blurred_img_tensor).unsqueeze(0).to(self.device).detach())

            image_embed = self.clip_model.encode_image(self.preprocess(img_tensor).unsqueeze(0).to(self.device).detach())
            image_size = torch.tensor(image.size[::-1])  # (w, h) -> (1, h, w)
            all_class_boxes, all_text_embeds, all_owlvit_scores = [], [], []

            # Compute top-k to speed up the inference process with XCLIP-v2
            if self.clip_topk > 0:
                image_labels_similarity = image_embed @ self.label_embeds.T
                _, topk_idxs = image_labels_similarity.topk(self.clip_topk, dim=1)
                topk_idxs_list = topk_idxs[0].tolist()
                clip_topk_preds = [name for idx, name in self.idx2name.items() if idx in topk_idxs_list]
                gt_label = -1 if gt_class_name not in clip_topk_preds else clip_topk_preds.index(gt_class_name)
            else:
                clip_topk_preds = None
                gt_label = owlvit_results["label"]

            if hasattr(self, "indexes_to_keep"):
                for class_name, boxes_pred in boxes_pred_dict.items():
                    boxes_pred["scores"] = [boxes_pred["scores"][i] for i in self.indexes_to_keep]
                    boxes_pred["boxes"] = [boxes_pred["boxes"][i] for i in self.indexes_to_keep]
                    boxes_pred["labels"] = list(range(len(self.indexes_to_keep))) #self.indexes_to_keep.copy() # prevent using the same memory address

            if self.add_allaboutbirds_descs and hasattr(self, "add_allaboutbirds_descs"):
                for class_name, boxes_pred in boxes_pred_dict.items():
                    boxes_pred["scores"].append(1.0)
                    boxes_pred["scores"].append(1.0)
                    boxes_pred["scores"].append(1.0)

                    boxes_pred["boxes"].append([0,0,w,h])
                    boxes_pred["boxes"].append([0,0,w,h])
                    boxes_pred["boxes"].append([0,0,w,h])

                    boxes_pred["labels"].append(len(self.indexes_to_keep))
                    boxes_pred["labels"].append(len(self.indexes_to_keep) + 1)
                    boxes_pred["labels"].append(len(self.indexes_to_keep) + 2)

            for class_name, boxes_pred in boxes_pred_dict.items():
                if clip_topk_preds is not None and class_name not in clip_topk_preds:
                    continue

                # Threshold is set to max if all boxes have confidence scores <= self.owl_vit_threshold
                threshold = self.owlvit_threshold if torch.any(torch.tensor(boxes_pred["scores"]) >= self.owlvit_threshold) else max(boxes_pred["scores"])
                threshold = 0.
                class_boxes, owlvit_scores, descs = [], [], []
                
                for box, score, desc in zip(boxes_pred["boxes"], boxes_pred["scores"], self.templated_descriptions[class_name]):
                    
                    if score >= threshold:
                        box[0] = max(0, min(box[0], image_size[1]))  # width (x)
                        box[1] = max(0, min(box[1], image_size[0]))  # height (y)
                        box[2] = max(0, min(box[2], image_size[1]))  # width (max_x)
                        box[3] = max(0, min(box[3], image_size[0]))  # height (max_y)

                        if hasattr(self, "blur_background") and self.blur_background:
                            kernel_size = self.blur_kernel if hasattr(self, "blur_kernel") else 7
                            box_img = gaussian_blur(img_tensor, torch.tensor(box).unsqueeze(0).int(), kernel_size=kernel_size).squeeze(0)
                        else:
                            
                            box_img = img_tensor[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            
                            # image = Image.open(owlvit_results["image_path"]).convert('RGB')
                            # shape = [(box[0], box[1]), (box[2], box[3])]
                            # image1 = ImageDraw.Draw(image)  
                            # image1.rectangle(shape, outline ="black", width=5)
                            # # image1.ellipse(shape, outline = 'black', width=5)
                            # img_tensor = self.pil2tensor(image)
                            # box_img = img_tensor
    
                        box_instance = self.preprocess(box_img)
                        class_boxes.append(box_instance)
                        owlvit_scores.append(score)
                        
                        descs.append(desc)
                    
                    for i, desc in enumerate(descs):
                        if len(desc) >= 250:
                            descs[i] = desc[:250]

                all_class_boxes.append(self.clip_model.encode_image(torch.stack(class_boxes).to(self.device)).cpu())
                all_owlvit_scores.append(torch.tensor(owlvit_scores))
                tokens = self.tokenizer(descs)
                text_embeds = F.normalize(self.clip_model.encode_text(tokens.to(self.device)))
                all_text_embeds.append(text_embeds.cpu())

                # PEIJIE: merge with the above loop
                # tokens = self.tokenizer([descriptor for idx, descriptor in enumerate(self.templated_descriptions[class_name])
                #                         if boxes_pred["scores"][idx] >= threshold])
                # text_embeds = F.normalize(self.clip_model.encode_text(tokens.to(self.device)))
                # all_text_embeds.append(text_embeds.cpu())

            # Each set of descriptors lead to different boxes                
            box_embeds = self.construct_padded_embeddings(all_class_boxes)
            text_embeds = self.construct_padded_embeddings(all_text_embeds)
            owlvit_scores = self.construct_padded_embeddings(all_owlvit_scores)

        return image_embed.detach().cpu() if blurred_image_embed is None else blurred_image_embed.detach().cpu(), \
               box_embeds, text_embeds, owlvit_scores, gt_label, owlvit_results["image_id"], clip_topk_preds

    def __len__(self):
        return len(self.image_ids)

    def construct_padded_embeddings(self, batch_embeddings):
        batch_padded_embeddings = []
        max_batch = self.max_num_descriptors    # max([embeddings.shape[0] for embeddings in batch_embeddings])
        
        if len(batch_embeddings[0].shape) == 2:
            emb_dim = batch_embeddings[0].shape[-1]
            for segment_embs in batch_embeddings:
                if len(segment_embs) < max_batch:
                    zeros = torch.zeros(max_batch, emb_dim)
                    zeros[:len(segment_embs)] = segment_embs
                    segment_embs = zeros
                else:
                    segment_embs = segment_embs[:max_batch]

                batch_padded_embeddings.append(segment_embs)
        else:
            for segment_embs in batch_embeddings:
                if len(segment_embs) < max_batch:
                    zeros = torch.zeros(max_batch)
                    zeros[:len(segment_embs)] = segment_embs
                    segment_embs = zeros
                else:
                    segment_embs = segment_embs[:max_batch]

                batch_padded_embeddings.append(segment_embs)

        return torch.stack(batch_padded_embeddings).detach()

