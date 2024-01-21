# reasoning
bird reasoning (size + habitat)

![Figure 1-1](./utils/project_description.png)

## Demo

```
python demo_image2text/run.py
```
## CNN based model
```
cd cnn_habitat_reaasoning/
```
### Train CUB
```
python cub_fintuning.py
```
### Train NABirds
```
python nabirds_fintuning.py
```

## CLIP

```
cd plain_clip/
```

## Keypoint detection and filtering
```
cd keypoint_detection/
python show_cub_segmentation.py
python show_cub_kp_segment.py
```