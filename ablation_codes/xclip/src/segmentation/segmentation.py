import os
from typing import Callable

import numpy as np
import torch
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from torchvision import transforms
from torchvision.ops import masks_to_boxes

import mmcv
import cv2
from mmcv.runner import load_checkpoint
from utils import PROJECT_ROOT


class SegmentationModel(object):
    def __init__(self, config, checkpoint=None, device='cuda:0', cfg_options=None):
        if not (config and checkpoint):
            config='mmdetection/configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic.py'
            checkpoint='../data/models/mask2former/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth'
            print("Segmentation model not specified, load default model...")
            
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.cfg_options = cfg_options
        self._load_detector(
            config, checkpoint, device, cfg_options)

    def _load_detector(self, config: str, checkpoint: str = None, device: str = 'cuda:0', cfg_options=None):
        # Load the config
        config = mmcv.Config.fromfile(config)
        # Update the test config options
        test_cfg=dict(
            panoptic_on=True,
            # For now, the dataset does not support evaluating semantic segmentation metric.
            semantic_on=False,
            instance_on=True,
            # max_per_image is for instance segmentation.
            max_per_image=100,
            iou_thr=0.1,
            # In Mask2Former's panoptic postprocessing, it will filter mask area where score is less than 0.5 .
            filter_low_score=False)
        config.model.test_cfg = test_cfg
        # 134
        # num_things_classes = 1000
        # num_stuff_classes = 1000
        # num_classes = num_things_classes + num_stuff_classes
        # config.num_classes = num_classes
        # config.num_things_classes = num_things_classes
        # config.num_stuff_classes = num_stuff_classes
        
        # config.model.panoptic_head.num_things_classes = num_things_classes
        # config.model.panoptic_head.num_stuff_classes = num_stuff_classes
        # config.model.panoptic_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]
        
        # Initialize the detector
        model = build_detector(config.model)
        # Load checkpoint
        checkpoint = load_checkpoint(model, checkpoint, map_location=device)
        # Set the classes of models for inference
        # * the default classes setting have 133 different classes, we can set it to any number. change to 2000.
        # model.CLASSES = checkpoint['meta']['CLASSES']
        model.CLASSES = list(range(2000))
        # We need to set the model's cfg for inference
        model.cfg = config
        # Convert the model to GPU
        model.to(device)
        # Convert the model into evaluation mode
        model.eval()
        self.model = model

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        return inference_detector(self.model, image)

    def show(self, image, result, out_file=None):
        show_result_pyplot(self.model, image, result, self.model.CLASSES, out_file=out_file)

    def save(self, image, result, path):
        mmcv.imwrite(image, path)
        mmcv.imwrite(result, path.replace('.jpg', '_segmentation.jpg'))

    # def __del__(self):
    #     del self.model


class SegMaskAndBox(object):

    @staticmethod
    def get_separated_segmentations(segmentation: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        # seg_types = set(segmentation.reshape(-1).tolist())
        # return torch.tensor([torch.tensor(segmentation == seg).int() for seg in seg_types]), seg_types
        seg_types = np.unique(segmentation)[::-1]
        segms = (segmentation[None] == seg_types[:, None, None])
        return torch.tensor(segms).int(), seg_types

    @staticmethod
    def get_segments_and_boxes(segmentation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        segments, seg_values = SegMaskAndBox.get_separated_segmentations(segmentation)
        # get bounding boxes for each segment
        # (x_min, y_min, x_max, y_max)
        boxes = masks_to_boxes(segments).int()
        # * some boxes are empty, so we need to remove them
        boxes = boxes[(boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])]
        return segments, boxes, seg_values
    
    # split disjoint segments into separate segments using openCV
    @staticmethod
    def split_disjoint_segments(segments: torch.Tensor, boxes: torch.Tensor, seg_values: np.ndarray, seg_results: dict) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        # convert segments to numpy array
        segments = segments.numpy()
        # Find countours for each segment
        contours = [cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for segment in segments]
        # Get bounding boxes for each contour
        box_list = [cv2.boundingRect(contour) for contour in contours]
        # Get new segments from contours with the corresponding seg_values
        new_segments = [cv2.drawContours(np.zeros_like(segment), contour, -1, 1, -1) for segment, contour in zip(segments, contours)]
        

class PreprocessInstances(object):
    def __init__(self, preprocess: Callable, segment_type: str = 'masked_box'):
        """_summary_

        Args:
            preprocess (Callable): preprocess function, takes a tensor and returns a tensor. The function should include resize, normalization, and any other preprocessing steps.
            segment_type (str, optional): Defaults to 'mask'. The type of instance to preprocess. Must be either 'mask' or 'box'.
        """
        self.preprocess = preprocess
        self.segment_type = segment_type
        self.preprocess.transforms.insert(0, transforms.ToPILImage())

    # segments and boxes can be derived from SegMaskAndBox module
    # boxes should in (x_min, y_min, x_max, y_max)
    def __call__(self, image: torch.Tensor, segments: torch.Tensor, boxes: torch.Tensor, mask_background: str = "white") -> torch.Tensor:
        if self.segment_type == "mask":
            instances = torch.stack([self.preprocess(image * mask) for mask in segments])

        elif self.segment_type == "box":
            instances = torch.stack([self.preprocess(image[:, y: y_max:, x: x_max]) for (x, y, x_max, y_max) in boxes.tolist()])

        elif self.segment_type == "masked_box_white":
            # create white background
            seg_backgrounds = [(torch.ones_like(segment) - segment)*255 for segment in segments]
            # create masked boxes (add white background to image)
            instances = torch.stack([image * mask + background for mask, background in zip(segments, seg_backgrounds)])
            instances = torch.stack([self.preprocess(instances[i, :, y: y_max:, x: x_max]) for i, (x, y, x_max, y_max) in enumerate(boxes)])

        elif self.segment_type == "masked_box_black":
            instances = torch.stack([image * mask for mask in segments])
            instances = torch.stack([self.preprocess(instances[i, :, y: y_max:, x: x_max]) for i, (x, y, x_max, y_max) in enumerate(boxes)])

        elif self.segment_type == "masked_box_gray":
            # CLIP transform mean (0.48145466, 0.4578275, 0.40821073)
            # CLIP transform std (0.26862954, 0.26130258, 0.27577711)
            
            # create mean background based on CLIP mean (three channels)
            gray_image = torch.ones_like(image) * torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            seg_backgrounds = [(torch.ones_like(image) - segment)*gray_image for segment in segments] # masked background
            # create masked boxes (add mean background to image)
            instances = torch.stack([image * mask + background for mask, background in zip(segments, seg_backgrounds)])
            instances = torch.stack([self.preprocess(instances[i, :, y: y_max:, x: x_max]) for i, (x, y, x_max, y_max) in enumerate(boxes)])
        
        else:
            raise ValueError('Instance type must be {"mask", "box" or "masked_box"}')

        return instances


class MMdetModelZoo(object):
    CONFIG_ROOT = f'{PROJECT_ROOT}/mmdetection/configs/mask2former'
    CHECKPOINT_ROOT = f'{PROJECT_ROOT}/data/models/mask2former/'
    model_zoo = {'rn50': {'config': os.path.join(CONFIG_ROOT, 'mask2former_r50_lsj_8x2_50e_coco-panoptic.py'),
                          'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth')},
                 
                 'rn101': {'config': os.path.join(CONFIG_ROOT, 'mask2former_r101_lsj_8x2_50e_coco-panoptic.py'),
                           'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_r101_lsj_8x2_50e_coco-panoptic_20220329_225104-c54e64c9.pth')},
                 
                 'swin_t': {'config': os.path.join(CONFIG_ROOT, 'mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic.py'),
                            'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic_20220326_224553-fc567107.pth')},
                 
                 'swin_s': {'config': os.path.join(CONFIG_ROOT, 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco-panoptic.py'),
                            'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco-panoptic_20220329_225200-c7b94355.pth')},
                 
                 'swin_b_1k': {'config': os.path.join(CONFIG_ROOT, 'mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco-panoptic.py'),
                               'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_swin-b-p4-w12-384_lsj_8x2_50e_coco-panoptic_20220331_002244-c149a9e9.pth')},
                 
                 'swin_b_21k': {'config': os.path.join(CONFIG_ROOT, 'mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic.py'),
                                'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth')},
                 
                 'swin_l_21k': {'config': os.path.join(CONFIG_ROOT, 'mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py'),
                                'checkpoint': os.path.join(CHECKPOINT_ROOT, 'mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth')},
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> tuple[str, str]:
        if model_name.lower() in cls.model_zoo.keys():
            return cls.model_zoo[model_name.lower()]['config'], cls.model_zoo[model_name.lower()]['checkpoint']
        else:
            raise ValueError(f'Model {model_name} not found in model zoo: {cls.model_zoo.keys()}')
    
    @classmethod
    def print_model_zoo(cls):
        print(f'Model zoo: {cls.model_zoo.keys()}')

    
    
