import numpy as np
import torch

from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from segmentation import get_pre_define_colors, draw_text, Drawer
from matching_backbone.models import load_model
from utils import load_descriptions


dataset = "cub"
descriptors = "chatgpt"
model_name = "owlvit-large-patch14"
processor = OwlViTProcessor.from_pretrained(f"google/{model_name}")
model = OwlViTForObjectDetection.from_pretrained(f"google/{model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load CLIP model
clip_model, preprocess, tokenizer = load_model(model_name="ViT-B/32")

# class_name = "Barn Swallow"
# image_path = "/home/lab/datasets/cub/CUB_200_2011/images/136.Barn_Swallow/Barn_Swallow_0034_130099.jpg"

# class_name = "Black-footed Albatross"
# image_path = "/home/lab/datasets/cub/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0049_796063.jpg"

# class_name = "Acadian Flycatcher"
# image_path = "/home/lab/datasets/cub/CUB_200_2011/images/037.Acadian_Flycatcher/Acadian_Flycatcher_0049_795580.jpg"

class_name = "Acadian Flycatcher"
image_path = "/home/lab/datasets/cub/CUB_200_2011/images/037.Acadian_Flycatcher/Acadian_Flycatcher_0043_29115.jpg"

image = Image.open(image_path).convert('RGB')

descriptions_only, _ = load_descriptions(dataset_name=dataset, prompt_type=0, desc_type=descriptors)
templated_descriptions, _ = load_descriptions(dataset_name=dataset, prompt_type=5, desc_type=descriptors)

descriptions_only = {key: [value.replace("It has", "").replace("It is", "").strip() for value in values]
                           for key, values in descriptions_only.items()}
templated_descriptions = {key: [value.replace("It has", "").replace("It is", "").strip() for value in values]
                          for key, values in templated_descriptions.items()}

class_list = list(descriptions_only.keys())
gt_label = class_list.index(class_name)

all_descriptions = list(descriptions_only.values())
n_descriptors = np.cumsum(np.array([len(descriptions_only[class_name]) for class_name in class_list]))

if descriptors == "chatgpt":
    all_descriptions = [[descriptor.split(":")[0] for descriptor in descriptors if ":" in descriptor] for descriptors in descriptions_only.values()]

with torch.no_grad():
    inputs = processor(text=all_descriptions, images=image, padding="max_length", truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]]).to(device)

# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0)

scores = torch.sigmoid(outputs.logits)
topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)

for i, (pred, idxs, scores) in enumerate(zip(results, topk_idxs.squeeze(1), topk_scores.squeeze(1))):
    # Only use the boxes returned from post-processing, scores & labels are recomputed for our own purposes
    boxes = pred['boxes'][idxs]
    boxes = boxes.detach().cpu().tolist()[n_descriptors[gt_label]: n_descriptors[gt_label + 1]]
    scores = scores.detach().cpu().tolist()[n_descriptors[gt_label]: n_descriptors[gt_label + 1]]
    labels = list(range(len(scores)))

    if len(boxes) > 0:
        for box, score, label in zip(boxes, scores, labels):
            print(f"Detected {descriptions_only[class_name][label]} with confidence {round(score, 3)} at location {box}")

        max_label = int(max(labels) if len(labels) > 0 else 0)
        text_colors = get_pre_define_colors(len(boxes), cmap_set=None)[: max_label + 1]

        # Prepare list of texts and colors for visualization
        text_list = [f"{idx + 1}. {descriptions_only[class_name][label]} | {round(score, 2)}" for idx, (label, score) in enumerate(zip(labels, scores))]
        text_colors = [text_colors[label] for label in labels]
        box_tags = [str(tag) for tag in list(range(1, len(labels) + 1))]

        out_image = Drawer.draw_boxes(image, boxes, text_colors, width=3, tags=box_tags, loc="below")
        out_image = draw_text(out_image, text_list=text_list + ["", f"Ground truth: {class_name}", f"Image path: {image_path}"], text_color=text_colors + [(0, 0, 0)] * 3)
        out_image.save(f'../results/{dataset}_{descriptors}_{image_path.split("/")[-1]}', quality=100, subsampling=0, dpi=(1024, 1024))

    else:
        print("No boxes found for the image")


