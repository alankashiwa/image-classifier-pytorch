import argparse
from PIL import Image
from collections import OrderedDict
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms, models

parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('image_path', type=str, help='image path')
parser.add_argument('chkpt_path', type=str, help='checkpoint path')
parser.add_argument('--top_k', type=int, help='return top k results', default=1)
parser.add_argument('--category_names', type=str, help='class name dict', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', help='use gpu')

args = parser.parse_args()

# Model builder
def build_model(chkpt_path):
    checkpoint = torch.load(chkpt_path)
    # The model to build
    architecture = checkpoint['transfer_model']
    # Classifier parameters
    cls_para = checkpoint['classifier']
    if architecture == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif architecture == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    else:
        raise ValueError('checkpoint not a vgg model')

    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(cls_para['input_layer'], cls_para['hidden_layer'][0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(cls_para['hidden_layer'][0], cls_para['hidden_layer'][1])),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(cls_para['hidden_layer'][1], cls_para['output_output_layer'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# Build the model from checkpoint
model = build_model(args.chkpt_path)

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open and resize the image
    im = Image.open(image)

    # Define the transform functions
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    return transform(im)


# Reverse the class_to_idx dictionary
idx_to_class = {v: k for k, v in model.class_to_idx.items() }

# Image Prediction
def predict(image_path, model, topk=5):
    # Implement the code to predict the class from an image file
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        output = model.forward(process_image(image_path).resize_(1, 3, 224, 224).to('cuda'))
        ps = torch.exp(output).cpu()
        probs, indices = ps.topk(topk)
        return [p for p in probs[0].numpy()], [idx_to_class[idx] for idx in indices[0].numpy()]


# Label Mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Predict and get the class names
probs, classes = predict(args.image_path, model, args.top_k)
class_names = [cat_to_name[cls] for cls in classes]
for i in range(len(probs)):
    print('Probability: {}, Identified as: {}'.format(probs[i], class_names[i]))