# %%
import torch
from datasets import load_dataset
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

# instructblip-flan-t5-xl
# instructblip-vicuna-7b
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_4bit=False, torch_dtype=torch.bfloat16).to('cuda:7')
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl",)

# %%
datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "test"),
    ("Matthijs/snacks", None, "validation"),
    ("rokmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train")
]

prompt1 = "describe this image in full detail. describe each and every aspect of the image so that an artist could recreate the image"
prompt2 = "create an extensive description of this image"
counter = 0
for name, config, split in datasets:
    d = load_dataset(name, config, split=split)
    for idx in range(len(d)):
        image = d[idx]["image"]
        desc = ""
        for _prompt in [prompt1, prompt2]:
            inputs = processor(images=image, text=_prompt, return_tensors="pt").to(model.device, torch.bfloat16)
            outputs = model.generate(**inputs, do_sample=False, num_beams=10, max_length=512, min_length=16, top_p=0.9, repetition_penalty=1.5, temperature=1)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            desc += generated_text + " "
        
        desc = desc.strip() # remove \n \t
        image.save(f"images/{counter}.jpg")
        print(counter, desc)

        with open("description.csv", "a") as f:
            f.write(f"{counter},{desc}\n")

        counter+=1
        torch.cuda.empty_cache()
# %%
