from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import time

def blip_captioning(image: Image):
    device = "cuda:7" if torch.cuda.is_available() else "cpu"

    t1 = time.time()
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)
    print("Time for loading model: ", time.time() - t1)

    t1 = time.time()
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    decoding_method = 'Beam search' # Nucleus sampling
    # generated_ids = model.generate(**inputs)
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        do_sample=decoding_method == 'Nucleus sampling',
        temperature=0.7,# 0.5->1
        length_penalty=1.0, # 1->2
        repetition_penalty=2.0, # 1-> 5.0
        max_length=50,
        min_length=1,
        num_beams=5,
        top_p=0.9)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print("Time for generating the text: ", time.time() - t1)

    return generated_text
