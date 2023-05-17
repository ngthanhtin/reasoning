from flask import Flask, render_template, request, redirect, url_for
from get_prediction import get_prediction
from generate_html import generate_html, generate_html_2
from torchvision import models
import json, os

app = Flask(__name__)

UPLOAD_FOLDER = './static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# mapping
imagenet_class_mapping = json.load(open('imagenet_class_index.json'))

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(weights='IMAGENET1K_V1')
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def get_image_class(path):
    # get_image(path)
    # path = get_path(path)
    images_with_tags = get_prediction(model, imagenet_class_mapping, path)
    generate_html(images_with_tags)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        get_image_class(path)
        return redirect(url_for('success', name=path.split('/')[-1]))

@app.route("/aifunction/", methods=['GET', 'POST'])
def move_forward():
    if request.form.get('clsBtn') == 'Classification':
        print("Classification is developing...")
        text = request.form['input_text']
        print('You entered: ', text)
        generate_html_2(text)
        return render_template('home_answer.html')
    if request.form.get('image2textBtn') == 'Image2Text':
        print("Image2Text is developing...")
        return render_template('index.html')


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    app.run(debug=True)
