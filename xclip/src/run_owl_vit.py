import argparse
import gc
import statistics
from datetime import datetime

import numpy as np
import spacy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from data_loader import DatasetWrapper, BoxWrapper
from matching_backbone.models import load_model
from segmentation import get_pre_define_colors, draw_text, Drawer
from utils import *

import torch

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import warnings
warnings.filterwarnings("ignore")

nlp_spacy = spacy.load("en_core_web_sm")


def get_clip_topk_predictions(args, clip_model, class_list, topk=10, batch_size=320, img_size=224):
    dataset = DatasetWrapper(dataset_name=args.dataset, transform=img_transform(img_size), distortion=f"{args.distortion}:{args.distortion_severity}",
                             samples_per_class=args.num_samples, random_seed=args.random_seed)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_topk_preds = {}
    label_embeds = compute_label_encodings(clip_model, class_list).to(device)
    clip_model = clip_model.to(device)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=f'Getting CLIP top-{topk} predictions', total=len(dataloader)):
            images, gt_labels, image_paths, image_ids, image_sizes = batch
            images = images.to(device)
            image_embs = clip_model.encode_image(images)
            image_embs = F.normalize(image_embs)

            image_labels_similarity = image_embs @ label_embeds.T
            batch_preds = image_labels_similarity.argsort(dim=1, descending=True).detach().cpu().tolist()
            for image_id, topk_preds in zip(image_ids, batch_preds):
                all_topk_preds[image_id] = [class_name for idx, class_name in enumerate(class_list) if idx in topk_preds[:topk]]

        gc.collect()
        torch.cuda.empty_cache()

    return all_topk_preds


def draw_boxes_with_descriptors_and_scores(image, image_path, class_name, logit, boxes, owlvit_scores, xclip_scores, labels, predict=False):
    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_colors = get_pre_define_colors(len(boxes), cmap_set=None)[: max_label + 1]

    # Prepare list of texts and colors for visualization
    if xclip_scores is not None:
        text_list = [f"{idx + 1}. {descriptions_only[class_name][label]} | {round(owlvit_score, 2)} | {round(xclip_score, 2)}" for idx, (label, owlvit_score, xclip_score) in enumerate(zip(labels, owlvit_scores, xclip_scores))]
    else:
        text_list = [f"{idx + 1}. {descriptions_only[class_name][label]} | {round(owlvit_score, 2)}" for idx, (label, owlvit_score) in enumerate(zip(labels, owlvit_scores))]

    text_colors = [text_colors[label] for label in labels]
    box_tags = [str(tag) for tag in list(range(1, len(labels) + 1))]

    out_image = Drawer.draw_boxes(image, boxes, text_colors, width=3, tags=box_tags, loc="below")
    out_image = draw_text(out_image, text_list=text_list + ["", f"{'Prediction' if predict else 'Ground truth'}: {class_name} (Logit: {round(logit, 2)})", f"Image path: {image_path}"], text_color=text_colors + [(0, 0, 0)] * 3)

    return out_image


def filter_owlvit_results(boxes_pred, owlvit_threshold):
    threshold = owlvit_threshold if torch.any(torch.tensor(boxes_pred["scores"]) >= owlvit_threshold) else max(boxes_pred["scores"])
    boxes, owlvit_scores, labels = [], [], []
    
    for box, owlvit_score, label in zip(boxes_pred["boxes"], boxes_pred["scores"], boxes_pred["labels"]):
        if owlvit_score >= threshold:
            boxes.append(box)
            owlvit_scores.append(owlvit_score)
            labels.append(label)

    return boxes, owlvit_scores, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='select model', default="owlvit-large-patch14", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument('--dataset', help='select dataset', default="imagenet", choices=["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "places365", "cub", "nabirds", "inaturalist2021"])
    parser.add_argument('--distortion', help='select distortion type if using ImageNet-C', default="defocus_blur", choices=["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "shot_noise", "gaussian_noise", "impulse_noise"])
    parser.add_argument('--distortion_severity', type=int, help='select distortion severity if using ImageNet-C', default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument('--batch_size', type=int, help='num batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=16)
    parser.add_argument('--num_samples', type=int, help='num images per class', default=-1)
    parser.add_argument('--device', help='select device', default="cuda:0", type=str)
    parser.add_argument('--random_seed', help='random seed (for data subsampling only)', default=42, type=int)

    parser.add_argument('--xclip_v3', help='run xclip_v3 using owlvit boxes', action="store_true")
    parser.add_argument('--clip_model', help='select clip model', default="ViT-B/32", type=str)
    parser.add_argument('--clip_topk', type=int, help='use clip topk predictions for faster inference', default=0)

    parser.add_argument('--descriptors', help='select descriptors for OwlViT', default="sachit", choices=["sachit", "chatgpt"])
    parser.add_argument('--prompt_type', type=int, help='select prompt type', default=5)

    parser.add_argument('--ablation', help='select a type for ablation study if specified', default="no_abs", choices=["no_abs", "boxes_only", "images_only"])
    parser.add_argument('--owlvit_threshold', type=float, help='select threshold for owl_vit', default=-1)
    parser.add_argument('--owlvit_conf_scores', help='use owlvit scores as confidence scores', action="store_true")

    parser.add_argument('--add_blur', help='add blur to image or background', type=str, default=None, choices=["image", "background"])
    parser.add_argument('--blur_kernel', help='blur kernel size', type=int, default=7)

    parser.add_argument('--visualize', help='visualization', action="store_true")
    parser.add_argument('--overwrite_boxes', help='save boxes for tuning', action="store_true")
    parser.add_argument('--test_class_list', help='a file containing a list of class to test', default=[], nargs='+')
    parser.add_argument('--check_box_files', help='check if box file exists (otherwise box computing will skip if the box folder exist.)', action="store_true")
    parser.add_argument('--verbose', help='print logs', action="store_true")

    # ---- 5 PARTS INFERENCE ----#
    parser.add_argument('--indexes_to_keep', nargs='+', type=int, default=list(range(12)), help='List of kept bird parts (only choose beak, belly, tail, wings, foots (1,2, 10, 8, 7))')
    parser.add_argument('--add_allaboutbirds_descs', help='Add Shape/Size/Habitat descs from AllaboutBirds', action="store_true")
    # ---- DRAWING INFERENCE ----#


    args = parser.parse_args()

    start_time = datetime.now()

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        check_device_availability(args.device)
        device = args.device

    num_parts = len(args.indexes_to_keep) + 3 if args.add_allaboutbirds_descs else len(args.indexes_to_keep)
    out_dir = f'{PROJECT_ROOT}/results/{args.dataset}/{args.descriptors}-{args.clip_model.replace("/", "")}/{args.ablation}/{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}_{str(num_parts)}parts'
    out_dir = f"{out_dir}_owl_vit"
    if args.visualize:
        out_dir = f"{out_dir}_viz"

    os.makedirs(out_dir, exist_ok=True)

    # Print and Save the results and configs
    with open(f'{out_dir}/configs.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    if args.visualize:
        os.makedirs(f"{out_dir}/correct", exist_ok=True)
        os.makedirs(f"{out_dir}/incorrect", exist_ok=True)

    if args.dataset == 'cub':
        boxes_dir = f"{PROJECT_ROOT}/pred_boxes/{args.dataset}/owl_vit_{args.model}_prompt_5_descriptors_{args.descriptors}"
    elif args.dataset == 'nabirds':
        boxes_dir = f"{PROJECT_ROOT}/pred_boxes/{args.dataset}/formated_data/"
    elif args.dataset == 'inaturalist2021':
        boxes_dir = f'{PROJECT_ROOT}/pred_boxes/{args.dataset}/425_classes_data/'
    elif args.dataset == "imagenet-c":
        boxes_dir = f"{boxes_dir}/distortion_{args.distortion}_severity_level_{args.distortion_severity}"

    # Prepare text embeddings
    descriptions_only, _ = load_descriptions(dataset_name=args.dataset, prompt_type=0, desc_type=args.descriptors)
    templated_descriptions, _ = load_descriptions(dataset_name=args.dataset, prompt_type=args.prompt_type, desc_type=args.descriptors)

    # Remove 'It has' and 'It is' from Sachit's CUB descriptors
    if args.dataset == "cub":
        descriptions_only = {key: [value.replace("It has", "").replace("It is", "").strip() for value in values]
                             for key, values in descriptions_only.items()}

        templated_descriptions = {key: [value.replace("It has", "").replace("It is", "").strip() for value in values]
                                  for key, values in templated_descriptions.items()}
    
    #---- sort the indexes to keep----#
    args.indexes_to_keep.sort()

    if args.add_allaboutbirds_descs:
        for class_name in descriptions_only.keys():
            descriptions_only[class_name] = [descriptions_only[class_name][i] for i in args.indexes_to_keep + [12,13,14]]
            templated_descriptions[class_name] = [templated_descriptions[class_name][i] for i in args.indexes_to_keep + [12,13,14]]
    else:
        for class_name in descriptions_only.keys():
            descriptions_only[class_name] = [descriptions_only[class_name][i] for i in args.indexes_to_keep]
            templated_descriptions[class_name] = [templated_descriptions[class_name][i] for i in args.indexes_to_keep]
    #----

    class_list = list(descriptions_only.keys())
    all_descriptions = list(descriptions_only.values())
    if args.descriptors == "chatgpt":
        all_descriptions = [[descriptor.split(":")[0] for descriptor in descriptors if ":" in descriptor] for descriptors in descriptions_only.values()]

    # get a list of target classes
    target_classes = check_file_or_list(args.test_class_list)
    target_classes = intersect_list(target_classes, class_list)

    # --------------------------------------------------------------------------------------------------------------
    # Index to Class Name Mapping
    # Assuming that the class names (from descriptors) are in the same order as the class indices (in the dataset)
    # --------------------------------------------------------------------------------------------------------------
    name2idx = {class_name: idx for idx, class_name in enumerate(class_list)}
    idx2name = dict(enumerate(class_list))
    assert len(name2idx) == len(idx2name) == len(class_list)
    target_classes = [name2idx[class_name] for class_name in target_classes]
    dataset_args = {"name2idx": name2idx, "idx2name": idx2name, "target_classes": target_classes, "device": device, "blur_kernel": args.blur_kernel, \
                    "dataset": args.dataset, "indexes_to_keep": args.indexes_to_keep, "add_allaboutbirds_descs": args.add_allaboutbirds_descs}
    if args.add_blur == "background":
        dataset_args['blur_background'] = True
    elif args.add_blur == "image":
        dataset_args['blur_image'] = True

    # --------------------------------------------------------------------------------------------------------------
    #   OWLVIT FOR LOCALIZATION
    # --------------------------------------------------------------------------------------------------------------
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if not os.path.exists(boxes_dir) or args.overwrite_boxes or args.check_box_files:
        os.makedirs(boxes_dir, exist_ok=True)

        # Load OwlViT model
        processor = OwlViTProcessor.from_pretrained(f"google/{args.model}")
        model = OwlViTForObjectDetection.from_pretrained(f"google/{args.model}")
        model.to(device)
        model.eval()

        # Load dataset and prepare data loader
        dataset = DatasetWrapper(dataset_name=args.dataset, transform=lambda x: processor(images=x, return_tensors='pt'),
                                 distortion=f"{args.distortion}:{args.distortion_severity}", samples_per_class=args.num_samples, random_seed=args.random_seed, **dataset_args)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        with torch.no_grad():
            text_inputs = processor(text=all_descriptions, padding="max_length", truncation=True, return_tensors="pt").to(device)
            total_descriptors = text_inputs['input_ids'].shape[0]
            text_inputs['input_ids'] = text_inputs['input_ids'].repeat(args.batch_size, 1)
            text_inputs['attention_mask'] = text_inputs['attention_mask'].repeat(args.batch_size, 1)

        for batch_idx, batch in tqdm(enumerate(dataloader), desc='Localizing descriptors', total=len(dataloader)):
            images, gt_labels, image_paths, image_ids, image_sizes = batch
            box_files_check = [os.path.exists(f"{boxes_dir}/{image_id}.pth") for image_id in image_ids]
            if all(box_files_check) and not args.overwrite_boxes:
                continue

            # Handle the last batch separately
            if batch_idx == len(dataloader) - 1:
                text_inputs['input_ids'] = text_inputs['input_ids'][:len(image_paths) * total_descriptors]
                text_inputs['attention_mask'] = text_inputs['attention_mask'][:len(image_paths) * total_descriptors]

            images['pixel_values'] = images['pixel_values'].squeeze(1).to(device)
            with torch.no_grad():
                owl_inputs = images | text_inputs
                owl_outputs = model(**owl_inputs)

                preds = processor.post_process_object_detection(owl_outputs, 0, image_sizes.to(device))
                scores = torch.sigmoid(owl_outputs.logits).detach().cpu()
                topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
                
                # PEIJIE: bound the boxes to the image size before we save the boxes in the future
                for i, (image_size, pred) in enumerate(zip(image_sizes, preds)):
                    # use the image size to set upper bound boxes
                    boxes_upper_bound = torch.tensor([0, 0, image_size[1], image_size[0]])
                    # replace negatives with zeros
                    bounded_boxes = torch.clamp(pred['boxes'].detach().cpu(), min=0)
                    # bound the boxes to the image size
                    bounded_boxes[:, 2:] = torch.min(bounded_boxes[:, 2:], boxes_upper_bound[2:])
                    preds[i]['boxes'] = bounded_boxes

            # Process OwlViT's output for each image in the batch separately
            for i, (pred, idxs, batch_scores) in tqdm(enumerate(zip(preds, topk_idxs.squeeze(1), topk_scores.squeeze(1))), total=args.batch_size):
                gt_label = gt_labels[i].item()
                gt_class_name = class_list[gt_label]

                # Only use the boxes returned from post-processing, scores & labels are recomputed for our own purposes
                batch_boxes = pred['boxes'][idxs]
                batch_scores = batch_scores.view(len(class_list), -1)
                batch_boxes = batch_boxes.view(len(class_list), -1, 4)

                # Construct a dictionary to save boxes prediction for each class's descriptors
                boxes_pred_dict = {}
                for class_name, scores, boxes in zip(class_list, batch_scores, batch_boxes):
                    n_descriptors = len(templated_descriptions[class_name])
                    scores = scores.tolist()[:n_descriptors]
                    boxes = boxes.tolist()[:n_descriptors]
                    
                    boxes_pred_dict[class_name] = {'scores': scores,
                                                   'boxes': boxes,
                                                   'labels': list(range(n_descriptors))}

                torch.save({"image_id": image_ids[i],
                            "image_path": image_paths[i],
                            "class_name": gt_class_name,
                            "label": gt_label,
                            "boxes_info": boxes_pred_dict}, f"{boxes_dir}/{image_ids[i]}.pth")

        model.cpu()
        del model
        gc.collect()
        # # with torch.cuda.device('cuda:6'):
        # torch.cuda.empty_cache()
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    #   CLIP-BASED CLASSIFICATION
    # --------------------------------------------------------------------------------------------------------------
    # Evaluation metric
    accuracy = []
    xclip_scores_logs = {}
    
    # Compute logit scores with X-CLIP (and OwlViT scores)
    if args.xclip_v3:
        # Load CLIP model
        clip_model, preprocess, tokenizer = load_model(model_name=args.clip_model, device=device)
        
        preprocess.transforms.insert(0, transforms.ToPILImage())
        
        clip_model = clip_model.to(device)
        clip_model.eval()

        # To replace the original CLIP transform
        clip_transform = transforms.Compose(
            [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
             transforms.CenterCrop((224, 224)),
             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
         ])
        
        # For using GPU in dataloader
        torch.multiprocessing.set_start_method('spawn', force=True)

        # Load dataset and prepare data loader
        dataset = BoxWrapper(boxes_dir=boxes_dir, clip_model=clip_model, preprocess=preprocess, templated_descriptions=templated_descriptions,
                             owlvit_threshold=args.owlvit_threshold, clip_topk=args.clip_topk, clip_tokenizer=tokenizer, **dataset_args)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        xclip_scores, logits = [], []
        for batch_idx, batch in tqdm(enumerate(dataloader), desc='Evaluating', total=len(dataloader)):
            image_embeds, box_embeds, text_embeds, owlvit_scores, gt_labels, image_ids, clip_topk_preds = batch
            clip_topk_preds = np.array(clip_topk_preds).T.tolist()

            # Note: batch_size may be smaller than args.batch_size for the last batch
            batch_size = box_embeds.shape[0]
            max_n_classes = box_embeds.shape[1]
            max_n_descriptors = box_embeds.shape[2]
            embed_dim = box_embeds.shape[-1]

            image_embeds = image_embeds.unsqueeze(1).repeat(1, 1, max_n_descriptors, 1).repeat(1, max_n_classes, 1, 1)

            if args.ablation == "boxes_only":
                box_embeds = F.normalize(box_embeds.to(device), dim=-1)
            elif args.ablation == "images_only":
                box_embeds = F.normalize(image_embeds.to(device), dim=-1)
            else:
                box_embeds = F.normalize(box_embeds.to(device) + image_embeds.to(device), dim=-1)
 
            # xclip_scores = box_embeds.view(batch_size, -1, embed_dim) @ torch.transpose(text_embeds.view(batch_size, -1, embed_dim).float(), dim0=-2, dim1=-1).to(device)
            xclip_scores = torch.einsum('bid,bjd->bij', box_embeds.float().view(batch_size, -1, embed_dim),
                                                        text_embeds.float().view(batch_size, -1, embed_dim).to(device))
            xclip_scores = torch.diagonal(xclip_scores, dim1=-2, dim2=-1).view(batch_size, max_n_classes, max_n_descriptors)

            if args.owlvit_conf_scores:
                logits = torch.sum(xclip_scores * torch.softmax(owlvit_scores, dim=-1).to(device), dim=-1) / torch.sum(xclip_scores != 0, dim=-1)
            else:
                logits = torch.sum(xclip_scores, dim=-1) / torch.sum(xclip_scores != 0, dim=-1)

            pred_labels = torch.argmax(logits, dim=-1)
            accuracy.extend((pred_labels.cpu() == gt_labels).int().tolist())
            print("Accuracy: {}/{} = {:.2f}".format(sum(accuracy), len(accuracy), statistics.mean(accuracy) * 100))

            if args.visualize:
                for idx, (pred_label, gt_label, image_id) in enumerate(zip(pred_labels, gt_labels, image_ids)):
                    owlvit_results = torch.load(f"{boxes_dir}/{image_id}.pth")
                    if args.dataset == 'cub':
                        owlvit_results["image_path"] = "/home/tin/datasets/cub/CUB/images/" + owlvit_results["image_path"].split("/")[-2] + "/" + owlvit_results["image_path"].split("/")[-1]
                    elif args.dataset == 'nabirds':
                        owlvit_results["image_path"] = owlvit_results["image_path"].replace('lab', 'tin')
                    elif args.dataset == 'inaturalist2021':
                        owlvit_results["image_path"] = "/home/tin/datasets/inaturalist2021_onlybird/bird_train/" + owlvit_results["image_path"].split("/")[-2] + "/" + owlvit_results["image_path"].split("/")[-1]

                    image_path = owlvit_results["image_path"]

                    boxes_pred_dict = owlvit_results["boxes_info"]
                    image = Image.open(image_path).convert('RGB')
                    w,h = image.size

                    for class_name, boxes_pred in boxes_pred_dict.items():
                        boxes_pred["scores"] = [boxes_pred["scores"][i] for i in args.indexes_to_keep]
                        boxes_pred["boxes"] = [boxes_pred["boxes"][i] for i in args.indexes_to_keep]
                        boxes_pred["labels"] = list(range(len(args.indexes_to_keep))) #args.indexes_to_keep.copy() # prevent using the same memory address

                    if args.add_allaboutbirds_descs:
                        for class_name, boxes_pred in boxes_pred_dict.items():

                            boxes_pred["scores"].append(1.0)
                            boxes_pred["scores"].append(1.0)
                            boxes_pred["scores"].append(1.0)

                            boxes_pred["boxes"].append([0,0,w,h])
                            boxes_pred["boxes"].append([0,0,w,h])
                            boxes_pred["boxes"].append([0,0,w,h])

                            boxes_pred["labels"].append(len(args.indexes_to_keep))
                            boxes_pred["labels"].append(len(args.indexes_to_keep)+1)
                            boxes_pred["labels"].append(len(args.indexes_to_keep)+2)

                    sub_dir = "correct" if pred_label == gt_label else "incorrect"
                    pred_class_name = class_list[pred_label] if args.clip_topk <= 0 else clip_topk_preds[idx][pred_label]
                    gt_class_name = class_list[gt_label] if args.clip_topk <= 0 else clip_topk_preds[idx][gt_label]

                    boxes, gt_owlvit_scores, labels = filter_owlvit_results(boxes_pred_dict[gt_class_name], args.owlvit_threshold)
                    gt_xclip_scores = xclip_scores[idx][gt_label].tolist()[:len(labels)] if args.xclip_v3 else None
                    gt_logit = logits[idx][gt_label].item() if args.xclip_v3 else sum(gt_owlvit_scores)
                    gt_out_image = draw_boxes_with_descriptors_and_scores(image, image_path, gt_class_name, gt_logit, boxes, gt_owlvit_scores, gt_xclip_scores, labels)

                    boxes, pred_owlvit_scores, labels = filter_owlvit_results(boxes_pred_dict[pred_class_name], args.owlvit_threshold)
                    pred_xclip_scores = xclip_scores[idx][pred_label].tolist()[:len(labels)] if args.xclip_v3 else None
                    pred_logit = logits[idx][pred_label].item() if args.xclip_v3 else sum(pred_owlvit_scores)
                    pred_out_image = draw_boxes_with_descriptors_and_scores(image, image_path, pred_class_name, pred_logit, boxes, pred_owlvit_scores, pred_xclip_scores, labels, predict=True)

                    result_img = Drawer.concat([gt_out_image, pred_out_image], horizontal=True)
                    result_img.save(f'{out_dir}/{sub_dir}/{wordify(gt_class_name.replace("/", "-"))}_{image_id}.jpg', quality=100, subsampling=0, dpi=(300, 300))

            for idx, image_id in enumerate(image_ids):
                xclip_scores_logs[image_id] = {"owlvit_scores": owlvit_scores[idx].tolist(),
                                               "xclip_scores": xclip_scores[idx].tolist(),
                                               "logits": logits[idx].tolist(),
                                               "gt_label": gt_labels[idx].item(),
                                               "clip_topk_preds": list(clip_topk_preds[idx])}

    else:
        for file_name in tqdm(sorted(os.listdir(boxes_dir))):
            owlvit_results = torch.load(f"{boxes_dir}/{file_name}")
            boxes_pred_dict = owlvit_results["boxes_info"]
            gt_label = owlvit_results["label"]

            logits = [statistics.mean(boxes_pred["scores"]) for class_name, boxes_pred in boxes_pred_dict.items()]
            pred_label = np.argmax(np.array(logits))
            pred_class_name = class_list[pred_label]
            accuracy.append(1 if pred_label == gt_label else 0)

            print("Accuracy: {}/{} = {:.2f}".format(sum(accuracy), len(accuracy), statistics.mean(accuracy) * 100))

    print("Accuracy: {}/{} = {:.2f}".format(sum(accuracy), len(accuracy), statistics.mean(accuracy) * 100))
    end_time = datetime.now()

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump({"Accuracy": statistics.mean(accuracy),
                   "Number of correct predictions": sum(accuracy),
                   "Number of examples": len(accuracy),
                   "Start time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                   "End time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                   "Duration": round((end_time - start_time).total_seconds() * 1.0 / 3600, 2)}, f, indent=4)

    with open(f'{out_dir}/xclip_scores_logs.json', 'w') as f:
        json.dump(xclip_scores_logs, f, indent=4)

