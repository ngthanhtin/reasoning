import re
import clip
import open_clip


# load model (currently clip) to get box-query scores
def load_model(model_name: str, device: str):
    if model_name in clip.clip._MODELS:
        model, transform = clip.load(model_name, device=device)
        tokenizer = clip.tokenize
    elif 'laion' in model_name:
        # from huggingface, the model card name has the following format: laion/CLIP-ViT-L-14-laion2B-s32B-b82K
        # where VIT-L-14 is the base model name, and laion2B-s32B-b82K is the training config
        pattern = r"(.*/)(.*?)-(.*?)-(.*?)-(.*?)-(.*)"
        matches = re.match(pattern, model_name)
        if matches:
            base_model_name = '-'.join(matches.group(3,4,5))
            training_config = matches.group(6)
        else:
            raise ValueError(f"model_name {model_name} is not in the correct format")
        model, training_transform, transform = open_clip.create_model_and_transforms(base_model_name, pretrained=training_config)
        tokenizer = open_clip.get_tokenizer(base_model_name)
    
    return model, transform, tokenizer
