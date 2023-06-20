import argparse
import torchmetrics
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from segmentation import Drawer, draw_segments
from data_loader import DatasetWrapper
from utils import *


def construct_text_embeddings(text_embeddings_dict):
    all_text_embeddings = []
    max_n_descriptors = max([len(class_desc_embs) for class_desc_embs in text_embeddings_dict.values()])
    emb_dim = list(text_embeddings_dict.values())[0].shape[-1]

    for class_name in list(text_embeddings_dict.keys()):
        if class_name == "corn_field":
            zeros = torch.zeros(max_n_descriptors, emb_dim)
            text_embs = zeros
        else:
            text_embs = text_embeddings_dict[class_name]

            if len(text_embs) < max_n_descriptors:
                zeros = torch.zeros(max_n_descriptors, emb_dim)
                zeros[:len(text_embs)] = text_embs
                text_embs = zeros
            else:
                text_embs = text_embs[:max_n_descriptors]

        all_text_embeddings.append(text_embs)

    return torch.stack(all_text_embeddings).to(device)


def construct_segment_embeddings(batch_segment_embeddings):
    batch_padded_segment_embeddings = []

    max_n_segments = max([segment_embs.shape[0] for segment_embs in batch_segment_embeddings])
    emb_dim = batch_segment_embeddings[0].shape[-1]

    for segment_embs in batch_segment_embeddings:
        if len(segment_embs) < max_n_segments:
            zeros = torch.zeros(max_n_segments, emb_dim)
            zeros[:len(segment_embs)] = segment_embs
            segment_embs = zeros
        else:
            segment_embs = segment_embs[:max_n_segments]

        batch_padded_segment_embeddings.append(segment_embs)

    return batch_padded_segment_embeddings


def plot_sachit_figure(args, img_idx, pred_label, gt_label, class_list, image_description_similarity=None, templated_descriptions=None, descriptions_mappings=None):

    plt.rcdefaults()
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(15.0, 5.0), dpi=100)

    # ----------------------------------- AX1 -----------------------------------
    label_descriptors = templated_descriptions[class_list[pred_label]]
    average_score = aggregate_similarity(image_description_similarity[pred_label][img_idx], dim=0).item() * 100
    descriptors, scores = [], []

    for templated_descriptor, score in sorted(zip(label_descriptors, image_description_similarity[pred_label][img_idx]), key=lambda x: x[1], reverse=True):
        scores.append(score.item() * 100)
        descriptor = descriptions_mappings[class_list[pred_label]][templated_descriptor]

        if len(descriptor) < 30:
            descriptors.append(descriptor)
        else:
            left = right = int(len(descriptor) / 2)
            while descriptor[left] != ' ':
                left -= 1
            while descriptor[right] != ' ':
                right += 1
            split_index = left if left <= right else right
            descriptors.append(descriptor[:split_index] + "\n" + descriptor[split_index:])

    scores = [average_score] + scores
    descriptors = ["Average"] + descriptors
    y_pos = np.arange(len(descriptors))

    ax1.barh(y_pos, scores, align='center', height=0.4, color='tab:blue')
    ax1.set_yticks(y_pos, labels=descriptors, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xticks([])

    for i, score in enumerate(scores):
        ax1.text(score - 8, i + 0.08, "{:.2f}".format(score), color='white', fontweight='bold')
    # ---------------------------------------------------------------------------

    # ----------------------------------- AX2 -----------------------------------
    label_descriptors = templated_descriptions[class_list[gt_label]]
    average_score = aggregate_similarity(image_description_similarity[gt_label][img_idx], dim=0).item() * 100
    descriptors, scores = [], []

    for k, v in sorted(zip(label_descriptors, image_description_similarity[gt_label][img_idx]), key=lambda x: x[1], reverse=True):
        descriptor = descriptions_mappings[class_list[gt_label]][k]
        scores.append(v.item() * 100)
        if len(descriptor) < 30:
            descriptors.append(descriptor)
        else:
            left = right = int(len(descriptor) / 2)
            while descriptor[left] != ' ':
                left -= 1
            while descriptor[right] != ' ':
                right += 1
            split_index = left if left <= right else right
            descriptors.append(descriptor[:split_index] + "\n" + descriptor[split_index:])

    scores = [average_score] + scores
    descriptors = ["Average"] + descriptors
    y_pos = np.arange(len(descriptors))

    ax2.barh(y_pos, scores, align='center', height=0.4, color='tab:red')
    ax2.set_yticks(y_pos, labels=descriptors, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xticks([])

    for i, score in enumerate(scores):
        ax2.text(score - 8, i + 0.08, "{:.2f}".format(score), color='white', fontweight='bold')

    ax2.set_title(f'GT: {class_list[gt_label]}', color='tab:red', loc='left')
    # ---------------------------------------------------------------------------

    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    fig.text(0.45, 0.93, "Our top prediction:", fontsize=18, color='black', ha='right')
    fig.text(0.50, 0.93, f' {class_list[pred_label]}', fontsize=18, color='tab:blue', ha='left')
    fig.text(0.25, 0.88, "and we say that because...", fontsize=14, color='black', ha='left')

    plt.tight_layout(pad=4.0)
    plt.box(False)

    tempfile = f"temp_{args.dataset}.jpg" if args.dataset != "imagenet-c" else f"temp_{args.dataset}_{args.distortion}_{args.distortion_severity}.jpg"
    fig.savefig(tempfile)
    plt.close()

    return Image.open(tempfile).convert('RGB')


def get_stats(batch_logits, sachit_batch_logits, gt_labels):
    stats = {"both_correct": [], "sachit_correct": [], "xclipv2_correct": [], "both_wrong": []}

    pred_labels = torch.argmax(batch_logits, dim=1)
    sachit_pred_labels = torch.argmax(sachit_batch_logits, dim=1)

    for pred_label, sachit_pred_label, gt_label in zip(pred_labels, sachit_pred_labels, gt_labels):
        if pred_label == sachit_pred_label == gt_label:
            stats["both_correct"].append(1)
        elif pred_label == gt_label:
            stats["xclipv2_correct"].append(1)
        elif sachit_pred_label == gt_label:
            stats["sachit_correct"].append(1)
        else:
            stats["both_wrong"].append(1)

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='select model', default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"])
    parser.add_argument('--dataset', help='select dataset', default="imagenet", choices=["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "places365", "cub"])
    parser.add_argument('--distortion', help='select distortion type if using ImageNet-C', default="defocus_blur", choices=["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "shot_noise", "gaussian_noise", "impulse_noise"])
    parser.add_argument('--distortion_severity', type=int, help='select distortion severity if using ImageNet-C', default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--image_emb_type', help='select types of embedding', default="mask2former_rn101_box")
    parser.add_argument('--batch_size', type=int, help='num batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=16)
    parser.add_argument('--num_samples', type=int, help='num images per class (-1 for all set)', default=-1)
    parser.add_argument('--prompt_type', type=int, help='select prompt type', default=5)
    parser.add_argument('--sachit', help='generate Sachit results for comparison', action="store_true")
    parser.add_argument('--visualize', help='visualization', action="store_true")
    parser.add_argument('--verbose', help='print logs', action="store_true")
    parser.add_argument('--imagenet_prompts', help='use 80 imagenet prompts by OpenAI', action="store_true")
    parser.add_argument('--full_img_emb', help='plus full image embedding to segment embeddings', action="store_true")
    parser.add_argument('--output_suffix', help='output suffix', default=None, type=str)
    parser.add_argument('--random_seed', help='random seed (for data subsampling only)', default=42, type=int)
    args = parser.parse_args()
    
    out_dir = f'{PROJECT_ROOT}/results/{args.dataset}/{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}'
    if args.output_suffix is not None:
        out_dir = f'{out_dir}_{args.output_suffix}'

    if args.visualize:
        out_dir = f"{out_dir}_viz"
        os.makedirs(out_dir, exist_ok=True)

        if args.sachit:
            visual_dirs = {"both_correct": "", "sachit_correct": "", "xclipv2_correct": "", "both_wrong": ""}
            for sub_dir in visual_dirs.keys():
                visual_dirs[sub_dir] = f"{out_dir}/explanations/{sub_dir}"
                os.makedirs(visual_dirs[sub_dir], exist_ok=True)
        else:
            visual_dir = f"{out_dir}/explanations/"
            os.makedirs(visual_dir, exist_ok=True)

    # Load model
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # Load dataset and prepare data loader
    dataset = DatasetWrapper(dataset_name=args.dataset, transform=preprocess, distortion=f"{args.distortion}:{args.distortion_severity}", samples_per_class=args.num_samples, random_seed=args.random_seed)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Check availability of precomputed image embeddings
    # image_embs_path = f"{PROJECT_ROOT}/segmentation/{args.dataset}/ViT-B_32_mask2former_rn101_box"

    model_name = args.model.replace("/", "_")
    image_embs_type = f"{model_name}_{args.image_emb_type}"
    image_embs_dir = f"{PROJECT_ROOT}/segmentation"
    image_embs_path = check_availability_of_precomputed_embs(image_embs_type, args.dataset, image_embs_dir)

    if args.dataset == "imagenet-c":
        image_embs_path = os.path.join(image_embs_path, args.distortion, str(args.distortion_severity))

    # Prepare text embeddings
    templated_descriptions, descriptions_mappings = load_descriptions(dataset_name=args.dataset, prompt_type=args.prompt_type)
    text_embeddings_dict = compute_description_encodings(model, templated_descriptions, templates=imagenet_templates if args.imagenet_prompts else None)
    text_embeddings = construct_text_embeddings(text_embeddings_dict)

    if args.sachit:
        sachit_templated_descriptions, sachit_descriptions_mappings = load_descriptions(dataset_name=args.dataset, prompt_type=1)
        sachit_text_embeddings_dict = compute_description_encodings(model, sachit_templated_descriptions, templates=imagenet_templates if args.imagenet_prompts else None)

    # Evaluation metric
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.n_classes).to(device)
    accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.n_classes, top_k=5).to(device)

    if args.sachit:
        sachit_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.n_classes).to(device)
        sachit_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.n_classes, top_k=5).to(device)

    stats = {"both_correct": [], "sachit_correct": [], "xclipv2_correct": [], "both_wrong": []}
    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Evaluating', total=len(dataloader)):
        images, labels, image_paths, image_ids, image_sizes = batch

        images = images.to(device)
        labels = labels.to(device)

        image_embs = model.encode_image(images)
        image_embs = F.normalize(image_embs)

        if args.sachit:
            image_description_similarity = [None] * dataset.n_classes
            image_description_similarity_cumulative = [None] * dataset.n_classes

            for i, (class_name, descriptors_embs) in enumerate(sachit_text_embeddings_dict.items()):
                dot_product_matrix = image_embs @ descriptors_embs.T.to(device)
                image_description_similarity[i] = dot_product_matrix
                image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i], dim=1)

            # create tensor of similarity means
            sachit_batch_logits = torch.stack(image_description_similarity_cumulative, dim=1)

        batch_logits = []

        for idx, image_id in enumerate(image_ids):
            image_data = torch.load(os.path.join(image_embs_path, f'{image_id}.pth'))
            if args.full_img_emb:
                segment_embs = F.normalize(image_data["image_embs"].to(device)) + image_embs[idx]
            else:
                segment_embs = F.normalize(image_data["image_embs"].to(device))
            
            dot_product_matrix = text_embeddings @ segment_embs.float().T
            clip_scores, mapping_indices = dot_product_matrix.max(dim=1)
            logits = torch.sum(clip_scores, dim=1).detach().cpu()
            batch_logits.append(logits)

            # Visualize the explanation for a single image
            if args.visualize:
                image = Image.open(image_paths[idx]).convert('RGB')
                segments, boxes = image_data['segments'], image_data['boxes']

                pred_idx = torch.argmax(logits, dim=0)
                pred_class = list(templated_descriptions.keys())[pred_idx]
                pred_descriptors = templated_descriptions[pred_class]

                pred_clip_scores = clip_scores[pred_idx]
                pred_texts = [descriptions_mappings[pred_class][pred_descriptors[idx]] for idx in mapping_indices[pred_idx]]
                pred_image = draw_segments(np.array(image), (segments, boxes), pred_texts, torch.tensor(range(len(pred_texts))), pred_clip_scores, emb_type=args.image_emb_type)

                gt_idx = labels[idx]
                gt_class = list(templated_descriptions.keys())[gt_idx]
                gt_descriptors = templated_descriptions[gt_class]

                gt_clip_scores = clip_scores[gt_idx]
                gt_texts = [descriptions_mappings[gt_class][gt_descriptors[idx]] for idx in mapping_indices[gt_idx]]
                gt_image = draw_segments(np.array(image), (segments, boxes), gt_texts, torch.tensor(range(len(gt_texts))), gt_clip_scores, emb_type=args.image_emb_type)

                pred_details = [[pred_text, "{:.2f}".format(score.item())] for pred_text, score in zip(pred_texts, pred_clip_scores)]
                gt_details = [[gt_text, "{:.2f}".format(score.item())] for gt_text, score in zip(gt_texts, gt_clip_scores)]

                result_img = Drawer.concat([image, Image.fromarray(pred_image), Image.fromarray(gt_image)], horizontal=True)
                result_img = Drawer.draw_text(result_img, ["image_id: {}".format(image_id),
                                                            "pred_class: {} --- pred_score: {:.2f}".format(pred_class, logits[pred_idx]),
                                                            "gt_class: {} --- gt_score: {:.2f}".format(gt_class, logits[gt_idx]),
                                                            "pred_details: " + "; ".join([" | ".join(triplet) for triplet in pred_details]),
                                                            "gt_details: " + "; ".join([" | ".join(triplet) for triplet in gt_details]),
                                                            "Note: phrase|segment_text_score"])

                result_img.save(f'{visual_dir}/{gt_class.split("/")[-1]}_{image_id}.jpg', quality=100, subsampling=0)

                if args.sachit:
                    # ------------------------------------------------------------------------------------------------------------------------
                    #   PLOT SACHIT RESULTS
                    # ------------------------------------------------------------------------------------------------------------------------
                    sachit_pred_idx = torch.argmax(sachit_batch_logits[idx], dim=0).item()
                    result_img_sachit = plot_sachit_figure(args=args, img_idx=idx,
                                                           pred_label=sachit_pred_idx, gt_label=gt_idx,
                                                           class_list=list(templated_descriptions.keys()),
                                                           image_description_similarity=image_description_similarity,
                                                           templated_descriptions=sachit_templated_descriptions,
                                                           descriptions_mappings=sachit_descriptions_mappings)
                    # ------------------------------------------------------------------------------------------------------------------------

                    final_img = Drawer.concat([result_img_sachit, result_img], horizontal=False)
                    if pred_idx == sachit_pred_idx == gt_idx:
                        root_dir = visual_dirs["both_correct"]
                    elif pred_idx == gt_idx:
                        root_dir = visual_dirs["xclipv2_correct"]
                    elif sachit_pred_idx == gt_idx:
                        root_dir = visual_dirs["sachit_correct"]
                    else:
                        root_dir = visual_dirs["both_wrong"]

                    final_img.save(f'{root_dir}/{gt_class.split("/")[-1]}_{image_id}.jpg', quality=100, subsampling=0)
                else:
                    result_img.save(f'{visual_dir}/{gt_class.split("/")[-1]}_{image_id}.jpg', quality=100, subsampling=0)

        batch_logits = torch.stack(batch_logits, dim=0)
        acc_top1 = accuracy_metric(batch_logits.to(device), labels)
        acc_top5 = accuracy_metric_top5(batch_logits.to(device), labels)

        if args.sachit:
            sachit_acc_top1 = sachit_accuracy_metric(sachit_batch_logits.to(device), labels)
            sachit_acc_top5 = sachit_accuracy_metric_top5(sachit_batch_logits.to(device), labels)

        batch_stats = get_stats(batch_logits, sachit_batch_logits, labels)
        for key in stats.keys():
            stats[key].extend(batch_stats[key])

        if args.verbose:
            print(f"Batch Top-1 and Top-5 accuracy: {100 * acc_top1} / {100 * acc_top5}")
            print(f"Batch Top-1 and Top-5 accuracy (SACHIT): {100 * sachit_acc_top1} / {100 * sachit_acc_top5}")

    accuracy_logs = {}
    accuracy_logs["Overall Top-1 Accuracy: "] = 100 * accuracy_metric.compute().item()
    accuracy_logs["Overall Top-5 Accuracy: "] = 100 * accuracy_metric_top5.compute().item()

    if args.sachit:
        accuracy_logs["Overall Top-1 Accuracy (SACHIT): "] = 100 * sachit_accuracy_metric.compute().item()
        accuracy_logs["Overall Top-5 Accuracy (SACHIT): "] = 100 * sachit_accuracy_metric_top5.compute().item()

    accuracy_logs["statistics"] = {"both_correct": len(stats['both_correct']),
                                   "both_wrong": len(stats['both_wrong']),
                                   "sachit_correct": len(stats['sachit_correct']),
                                   "xclipv2_correct": len(stats['xclipv2_correct'])}
    print(f"Statistics:\n "
          f"- both_correct: {len(stats['both_correct'])}\n - both_wrong: {len(stats['both_wrong'])}\n "
          f"- sachit_correct: {len(stats['sachit_correct'])}\n - xclipv2_correct: {len(stats['xclipv2_correct'])}")

    # Print and Save the results and configs
    os.makedirs(out_dir, exist_ok=True)

    for key, value in accuracy_logs.items():
        print(key, value)

    with open(f'{out_dir}/configs.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(accuracy_logs, f, indent=4)


