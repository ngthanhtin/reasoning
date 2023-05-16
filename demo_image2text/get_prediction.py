import json
import pandas as pd
import io
import glob
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_category(model, imagenet_class_mapping, image_path):
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    transformed_image = transform_image(image_bytes=image_bytes)
    outputs = model.forward(transformed_image)
    
    _, category = outputs.max(1)
    
    predicted_idx = str(category.item())
    return imagenet_class_mapping[predicted_idx]

def get_prediction(model, imagenet_class_mapping, file_path):
    image_with_tags = {}
    
    image_with_tags[file_path.split('/')[-1]] = get_category(model, imagenet_class_mapping, image_path=file_path)[1]
    return image_with_tags

