import statistics
import itertools
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import DatasetWrapper, BoxWrapper
import gc

from utils import *


def get_clip_topk_predictions(clip_model, class_list, topk=10, batch_size=16):
    dataset = DatasetWrapper(dataset_name="cub", transform=img_transform(224), samples_per_class=-1, random_seed=42)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_topk_preds = {}
    label_embeds = compute_label_encodings(clip_model, class_list)

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


def compute_mean_std_of_clip_matching_scores():
    input_dir = f"{PROJECT_ROOT}/results"

    # Sachit's descriptors
    '''
    result_dirs = {"cub": f"{input_dir}/cub/03_18_2023-14:46:41_owl_vit",
                   "imagenet-a": f"{input_dir}/imagenet-a/03_18_2023-14:46:46_owl_vit",
                   "cub-sachit": f"{input_dir}/cub/03_18_2023-18:08:35_owl_vit",
                   "imagenet-a-sachit": f"{input_dir}/imagenet-a/03_18_2023-18:08:43_owl_vit"}
    '''

    # ChatGPT's descriptors
    result_dirs = {
                   # "cub": f"{input_dir}/cub/part-based/03_21_2023-01:29:17_owl_vit",
                   # "imagenet-a": f"{input_dir}/imagenet-a/part-based/03_22_2023-13:07:42_owl_vit",
                   # "cub-threshold": f"{input_dir}/cub/part-based/03_22_2023-13:07:28_owl_vit_threshold",
                   # "imagenet-a-threshold": f"{input_dir}/imagenet-a/part-based/03_22_2023-13:07:35_owl_vit_threshold",
                   "cub-sachit": f"{input_dir}/cub/ablation_study/03_21_2023-02:21:09_owl_vit_images_only",
                   # "imagenet-a-sachit": f"{input_dir}/imagenet-a/ablation_study/"
                   }

    for dataset, result_dir in result_dirs.items():
        with open(f'{result_dir}/xclip_scores_logs.json', 'r') as input_file:
            xclip_scores_logs = json.load(input_file)

        all_logits = []
        for image_id, results in xclip_scores_logs.items():
            all_logits.extend(results["logits"])

        print("Dataset {}: Mean +/- STD = {:.2f} +/- {:.2f}".format(dataset, statistics.mean(all_logits), statistics.stdev(all_logits)))


def check_descriptors(desc_type, dataset):
    with open(f"{PROJECT_ROOT}/data/text/{desc_type}/descriptors_{dataset}.json") as input_file:
        all_descriptions = json.load(input_file)

    class_desc_lengths = [len(class_descriptions) for class_descriptions in all_descriptions.values()]
    print("Dataset {}: Max: {} --- Min: {} --- Mean +/- STD = {:.2f} +/- {:.2f}".format(dataset, max(class_desc_lengths), min(class_desc_lengths),
                                                                                        statistics.mean(class_desc_lengths), statistics.stdev(class_desc_lengths)))
    return list(all_descriptions.keys())


def compute_number_of_boxes_before_and_after_filtering():
    input_dir = f"{PROJECT_ROOT}/results"

    # ChatGPT's descriptors
    result_dirs = {
                   "cub": f"{input_dir}/cub/part-based/03_21_2023-01:29:17_owl_vit",                                    # Top50
                   "cub-threshold": f"{input_dir}/cub/part-based/03_22_2023-05:19:09_owl_vit_top50_threshold",          # Top50
                   "imagenet-a": f"{input_dir}/imagenet-a/part-based/03_22_2023-13:07:42_owl_vit",
                   "imagenet-a-threshold": f"{input_dir}/imagenet-a/part-based/03_22_2023-13:07:35_owl_vit_threshold",
                   }

    for dataset, result_dir in result_dirs.items():
        with open(f'{result_dir}/xclip_scores_logs.json', 'r') as input_file:
            xclip_scores_logs = json.load(input_file)

        no_discarded_boxes = []
        for image_id, results in xclip_scores_logs.items():
            no_discarded_boxes.append(results["xclip_scores"][0].count(0))

        print("Dataset {}: Mean +/- STD = {:.2f} +/- {:.2f}".format(dataset, statistics.mean(no_discarded_boxes), statistics.stdev(no_discarded_boxes)))


def compute_importance_scores_and_rank_bird_parts():
    input_dir = f"{PROJECT_ROOT}/results"

    # ChatGPT's descriptors
    result_dir = f"{input_dir}/cub/part-based/03_21_2023-01:29:17_owl_vit"
    parts = ["back", "beak", "belly", "breast", "crown", "forehead", "eyes", "legs", "wings", "nape", "tail", "throat"]

    with open(f'{result_dir}/xclip_scores_logs.json', 'r') as input_file:
        xclip_scores_logs = json.load(input_file)

    boxes_dir = f"{PROJECT_ROOT}/pred_boxes/cub/owl_vit_owlvit-large-patch14_descriptors_chatgpt"

    all_accuracies = {}

    descriptions_only, _ = load_descriptions(dataset_name="cub", prompt_type=0, desc_type="chatgpt")
    class_list = list(descriptions_only.keys())

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess.transforms.insert(0, transforms.ToPILImage())
    clip_model.eval()

    clip_topk_preds = get_clip_topk_predictions(clip_model, class_list, topk=50)

    for r in tqdm(range(len(parts)), desc='Ranking parts', total=len(parts)):
        part_combinations = list(itertools.combinations(parts, r+1))

        for part_combination in tqdm(part_combinations):
            part_ids = [part_combination.index(part) for part in part_combination]
            accuracy = []

            for image_id, results in xclip_scores_logs.items():
                logits = np.mean(np.array(results["xclip_scores"])[:, part_ids], axis=1)
                pred_label = np.argmax(logits)

                pred_boxes = torch.load(f"{boxes_dir}/{image_id}.pth")
                assert pred_boxes["image_id"] == image_id

                gt_label = -1 if pred_boxes["class_name"] not in clip_topk_preds[image_id] else clip_topk_preds[image_id].index(pred_boxes["class_name"])
                accuracy.append(1 if pred_label == gt_label else 0)

            all_accuracies["-".join(list(part_combination))] = statistics.mean(accuracy)

        all_accuracies = dict(sorted(all_accuracies.items(), key=lambda x: -x[1]))
        with open("../results/cub/part_analysis_results.json", "w") as output_file:
            json.dump(all_accuracies, output_file, indent=4)


if __name__ == '__main__':
    # compute_mean_std_of_clip_matching_scores()
    # compute_number_of_boxes_before_and_after_filtering()

    compute_importance_scores_and_rank_bird_parts()

    # class_names1 = check_descriptors(desc_type="chatgpt", dataset="imagenet")
    # class_names2 = check_descriptors(desc_type="sachit", dataset="imagenet")
    #
    # if class_names1 == class_names2:
    #     print("Two lists are identical")


