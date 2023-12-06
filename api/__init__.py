import base64
import binascii
import torch
import numpy as np

from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmfashion.utils import get_img_tensor
from mmfashion.models import build_predictor

from flask import Flask, request, jsonify

from .config import ApplicationConfig


app = Flask(__name__)

app.config.from_object(ApplicationConfig)

"""
from .classify import classify_blueprint

app.register_blueprint(classify_blueprint, url_prefix='/image') 
"""

@app.route("/classify", methods=['POST'])
def classify_image():   
    image_data = request.get_json()['image']

    try:
        # convert base64 to image
        image = base64.b64decode(image_data, validate=True)
    
        with open("pretty-girl-dress.jpg", "wb") as f:
            f.write(image)

        cfg = Config.fromfile('configs/attribute_predict/global_predictor_vgg_attr.py')
        img_tensor = get_img_tensor("pretty-girl-dress.jpg", False)

        landmark_tensor = torch.zeros(8)
        cfg.model.pretrained = None
        model = build_predictor(cfg.model)
        load_checkpoint(model, 'checkpoint/Predict/vgg/global/latest.pth', map_location='cpu')

        model.eval()

        attr_prob = model(img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
        
        classification = show_prediction(cfg.data.test, attr_prob)

        return jsonify(classification)

    except binascii.Error as e:
        error_msg = str(e)
        error_code = 400
        response = {
            'error': {
                'message': error_msg,
                'status_code': error_code
            }
        }

        return jsonify(response), error_code

def show_prediction(cfg, pred):
    attr_cloth_file = open(cfg.attr_cloth_file).readlines()
    attr_idx2name = {}
    for i, line in enumerate(attr_cloth_file[2:]):
        attr_idx2name[i] = line.strip('\n').split()[0]
    if isinstance(pred, torch.Tensor):
        data = pred.data.cpu().numpy()
    elif isinstance(pred, np.ndarray):
        data = pred
    else:
        raise TypeError('type {} cannot be calculated.'.format(type(pred)))
    
    result = dict()
    for i in range(pred.size(0)):
        indexes = np.argsort(data[i])[::-1]
        for idx in range(0, 10):
            result[attr_idx2name[indexes[idx]]] = float(data[i][indexes[idx]])
        
    return result


@app.get("/retrieve/<string:image>")
def retrieve_images(image):
    images = dict()

    with open("image_urls.txt") as f:
        for i, url in enumerate(f.readlines(), start = 1):
            images[i] = url.strip()

    return jsonify(images)