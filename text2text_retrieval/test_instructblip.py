# %%
from datasets import load_dataset
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from PIL import Image
import os, shutil

# %%
# instructblip-flan-t5-xl
# instructblip-vicuna-7b
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_4bit=False, torch_dtype=torch.bfloat16).to('cuda:7')
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl",)

# %%
class ImageFolderWithPaths(ImageFolder):

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        
        return (img, label ,path)
    
dataset = ImageFolderWithPaths('/home/tin/datasets/cub/CUB/train/',transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# %%
# prompt1 = "describe this image in full detail. describe each and every aspect of the image so that an artist could recreate the image"
# prompt2 = "create an extensive description of this image"
prompt1 = "describe the habitat of this image."
prompts = [prompt1]
counter = 0

if not os.path.exists('./images/'):
    os.makedirs("./images/")

for image, label, path in loader:
    image_name = path[0].split('/')[-1]
    desc = ""
    for _prompt in prompts:
        inputs = processor(images=image, text=_prompt, return_tensors="pt").to(model.device, torch.bfloat16)
        outputs = model.generate(**inputs, do_sample=True, num_beams=5, max_length=128, min_length=16, top_p=0.9, repetition_penalty=1.5, temperature=1)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        desc += generated_text + " "
    
    desc = desc.strip() # remove \n \t
    shutil.copy(path[0], f"images/{image_name}")
    # image.save(f"images/{image_name}.jpg")
    print(image_name, desc)

    with open("description.csv", "a") as f:
        f.write(f"{image_name},{desc}\n")

    counter+=1
    torch.cuda.empty_cache()
# %%
