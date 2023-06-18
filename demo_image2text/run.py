from flask import Flask, render_template, request, redirect, url_for
from get_prediction import get_prediction
from generate_html import generate_html, generate_html_2
from torchvision import models
import json, os

from PIL import Image
from image2text.blip2 import blip_captioning
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import clip
from image2text.image_text_matching.clip_matching import image_text_matching

app = Flask(__name__)

DEVICE = 'cpu'# 'cuda:7'
UPLOAD_FOLDER = './static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#----------INITIALIZE MODELS---------------#
# mapping
imagenet_class_mapping = json.load(open('imagenet_class_index.json'))

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
classification_model = models.densenet121(weights='IMAGENET1K_V1').to(DEVICE)
# Since we are using our model only for inference, switch to `eval` mode:
classification_model.eval()

# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
#     )
# blip2_model.to(DEVICE)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model.eval()
clip_model.requires_grad_(False)
# ----------------------------------------#

def get_image_class(path):
    # get_image(path)
    # path = get_path(path)
    images_with_tags = get_prediction(classification_model, imagenet_class_mapping, path)
    # generate_html(images_with_tags)
    return images_with_tags

def get_allaboutbirds_info(path="/home/tin/reasoning/scraping/allaboutbirds_ids/"):
    """
    return: a dict E.g. "Dark-eyed Junco": {
        "Size": "large small",
        "Color": "Dark-eyed Junco",
        "Behavior": "The Dark-eyed Junco is a medium-sized sparrow with a rounded head, a short, stout bill and a fairly long, conspicuous tail.",
        "Habitat": "Example query for example 1",
        },
    """
    birds = os.listdir(path)
    data = {}
    for bird in birds:
        data[bird] = {}
        # read meta file
        meta_json_path = os.path.join(path, f'{bird}/meta.json')

        f = open(meta_json_path)
        bird_data = json.load(f)
        
        # data[bird]['Size'] = bird_data['Size']['description']
        # data[bird]['Color'] = bird_data['Color']['description']
        # data[bird]['Behavior'] = bird_data['Behavior']['description']
        # data[bird]['Habitat'] = bird_data['Habitat']['description']

        data[bird] = ''
        data[bird] += f"Size: {bird_data['Size']['description']}\n"
        data[bird] += f"Color: {bird_data['Color']['description']}\n"
        data[bird] += f"Behavior: {bird_data['Behavior']['description']}\n"
        data[bird] += f"Habitat: {bird_data['Habitat']['description']}\n"

    return data




@app.route('/')
def home():
    examples = get_allaboutbirds_info()
    return render_template('home.html', data=examples)

    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # TODO: remove all current.* in static folder

        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        # path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file_type = file1.filename.split('.')[-1]
        print('haha')
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'current.{file_type}')
        
        file1.save(path)
        
        # get_image_class(path)
        
        return '', 204

@app.route("/aifunction/", methods=['GET', 'POST'])
def move_forward():
    
    if request.form.get('clsBtn') == 'Classification':
        image_files = os.listdir(UPLOAD_FOLDER)
        image_path = None
        for filename in image_files:
            if 'current' in filename:
                image_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not image_path:
            print("Can't find the image...")
            return '', 204
        else:
            image_with_tags = get_image_class(image_path)
            text = str(image_with_tags)
            generate_html_2(text)
            return render_template('home_answer.html')
    
    if request.form.get('image2textBtn') == 'Image2Text':
        image_files = os.listdir(UPLOAD_FOLDER)
        image_path = None
        for filename in image_files:
            if 'current' in filename:
                image_path = os.path.join(UPLOAD_FOLDER, filename)

        if not image_path:
            print("Can't find the image...")
            return '', 204
        else:
            image = Image.open(image_path)
            print("Progessing....")
            # text = blip_captioning(blip2_model, blip2_processor, image)
            text = request.form['input_text']
            texts, similarities = image_text_matching(clip_model, clip_preprocess, image, text)

            similarities, texts = zip(*sorted(zip(similarities, texts)))
            similarities, texts = similarities[::-1], texts[::-1]
            render_text = ''.join(f"{text}: {str(sim)} <br>" for text, sim in zip(texts, similarities))
    
            generate_html_2(render_text, image_path)
            return render_template('home_answer.html')


@app.route('/')
def success(name):
    return render_template('home_answer.html')


if __name__ == '__main__' :
    app.run(debug=True)
