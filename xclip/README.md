# xclip

### LAION CLIP 
To use LAION CLIP, install with:
```python
pip install open_clip_torch
```
The LAION CLIP model card can be found [here](https://huggingface.co/models?library=open_clip&sort=downloads&search=ViT-L%2F14).
(Copy the full title from the model card and parse it to `--clip_model`, i.e., `python src/run_owl_vit.py --clip_model "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"`)