import os
import argparse
import numpy as np

from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

from segmentation import MMdetModelZoo, SegmentationModel, PreprocessInstances, SegMaskAndBox
from data_loader import DatasetWrapper
from utils import *

import copy


class SaveSegmentEmbeddings(object):
    """Save segment embeddings to a file."""

    def __init__(self, out_folder: str, overwrite=False):
        self.out_folder = out_folder
        self.overwrite = overwrite

    def set_config(self, preprocess: Callable, segment_type='box'):
        self.preprocess = PreprocessInstances(preprocess, segment_type)
        self.segment_type = segment_type

    def __call__(self, segmentation_model: Callable, model: Callable, dataloader, device: str ='cuda:0'):
        """Save segment embeddings to a file."""
        pil2tensor = transforms.ToTensor()
        num_instances = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc='Saving segment embeddings', total=len(dataloader)):
                images, labels, image_paths, image_ids, image_sizes = batch
            
                for image_path, image_id in zip(image_paths, image_ids):
                    if self.overwrite or not os.path.exists(f'{self.out_folder}/{image_id}.pth'):
                        image = Image.open(image_path).convert('RGB')

                        # get segments
                        seg_results = segmentation_model(np.array(image))
                        segments, boxes, seg_values = SegMaskAndBox.get_segments_and_boxes(seg_results['pan_results'])
                        num_instances.append(len(seg_values))

                        # preprocess instances
                        img_tensor = pil2tensor(image)
                        img_instances = self.preprocess(img_tensor, segments, boxes)
                        image_embeddings = model.encode_image(img_instances.to(device)).detach().cpu()  # (n, embed_dim)
                        torch.save({"image_embs": image_embeddings,
                                    "segments": segments,
                                    "boxes": boxes,
                                    "seg_values": seg_values, }, f'{self.out_folder}/{image_id}.pth')
                        del seg_results
        avg_instances = np.mean(num_instances)
        print(f'Average number of instances per image: {avg_instances}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='select model', default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"])
    parser.add_argument('--dataset', help='select dataset', default="imagenet", choices=["imagenet", "imagenet-v2", "imagenet-a", "imagenet-c", "places365", "cub"])
    parser.add_argument('--distortion', help='select distortion type if using ImageNet-C', default="defocus_blur", choices=["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "shot_noise", "gaussian_noise", "impulse_noise"])
    parser.add_argument('--distortion_severity', type=int, help='select distortion severity if using ImageNet-C', default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--batch_size', type=int, help='num batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='num workers for batch processing', default=16)
    parser.add_argument('--segment_model', help='select segment model', default="mask2former", choices=["mask2former", "faster_rcnn", "owl_vit"])
    parser.add_argument('--segment_backbone', help='select config for segment model', default="rn101", choices=["rn101", "swin_b_1k", "swin_b_21k"])
    parser.add_argument('--segment_type', help='select segment type (not for owl_vit)', default="box", choices=["box", "masked_box_white", "masked_box_black", "masked_box_gray"])
    parser.add_argument('--overwrite', help='overwrite segments embeddings if existed', action="store_true")
    parser.add_argument('--out_dir', help='specify output directory', default=f"{PROJECT_ROOT}/segmentation")
    parser.add_argument('--device', default="cuda:0", type=str, help="Device name, i.e., 'cuda:0' or 'cpu' (default: 'cuda:1')")

    args = parser.parse_args()
    device = args.device
    
    # Load model
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # Load dataset and prepare data loader
    dataset = DatasetWrapper(dataset_name=args.dataset, transform=preprocess, distortion=f"{args.distortion}:{args.distortion_severity}")
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    out_folder = f"{args.out_dir}/{args.dataset}/{args.model.replace('/', '_')}_{args.segment_model}_{args.segment_backbone}_{args.segment_type}"
    if args.dataset == "imagenet-c":
        out_folder = f"{out_folder}/{args.distortion}/{args.distortion_severity}"
    os.makedirs(out_folder, exist_ok=True)

    # Initialize dataloader, image encoder, and segment model
    config, checkpoint = MMdetModelZoo.get_config(args.segment_backbone)
    segmentation_model = SegmentationModel(config=config, checkpoint=checkpoint, device=device)

    # Save image embeddings
    save_embeddings = SaveSegmentEmbeddings(out_folder=out_folder, overwrite=args.overwrite)
    save_embeddings.set_config(preprocess=copy.deepcopy(preprocess), segment_type=args.segment_type)
    save_embeddings(segmentation_model=segmentation_model, model=model, dataloader=dataloader, device=device)

