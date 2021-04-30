import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torchvision

from exceptions import NoSuchNameError , NoIndexError

def load_model(model_name):
    
    try:
        if '.pt' in model_name: #for saved model (.pt)
            if torch.typename(torch.load(model_name)) == 'OrderedDict':

                """
                if you want to use customized model that has a type 'OrderedDict',
                you shoud load model object as follows:
                
                from Net import Net()
                model=Net()
                """
                model.load_state_dict(torch.load(model_name))
            else:
                model = torch.load(model_name) 

        elif hasattr(models, model_name): #for pretrained model (ImageNet)
            model = getattr(models, model_name)(pretrained=True)

        model.eval()
        if cuda_available():
            model.cuda()
    except:
        raise ValueError(f'Not unvalid model was loaded: {model_name}')
        
    return model

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_image(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255

    return img

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if cuda_available():
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)

def save(mask, img, img_path, model_path):

    mask = (mask - np.min(mask)) / np.max(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    gradcam = 1.0 * heatmap + img
    gradcam = gradcam / np.max(gradcam)

    index = img_path.find('/')
    index2 = img_path.find('.')
    path = 'result/' + img_path[index + 1:index2] +'/'+model_path
    if not (os.path.isdir(path)):
        os.makedirs(path)

    gradcam_path = path + "/gradcam.png"
    cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))

    
def is_int(v):
    v = str(v).strip()
    return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()

def _exclude_layer(layer):

    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False

def choose_tlayer(model):
    name_to_num = {}
    num_to_layer = {}
    for idx, data in enumerate(model.named_modules()):        
        name, layer = data
        if _exclude_layer(layer):
            continue
        
        name_to_num[name] = idx
        num_to_layer[idx] = layer
        print(f'[ Number: {idx},  Name: {name} ] -> Layer: {layer}\n')
   
    print('\n<<-------------------------------------------------------------------->>')
    print('\n<<      You sholud not select [classifier module], [fc layer] !!      >>')
    print('\n<<-------------------------------------------------------------------->>\n')

    a = input(f'Choose "Number" or "Name" of a target layer: ')

    
    if a.isnumeric() == False:
        a = name_to_num[a]
    else:
        a = int(a)
    try:
        t_layer = num_to_layer[a]
        return t_layer    
    except IndexError:
        raise NoIndexError('Selected index (number) is not allowed.')
    except KeyError:
        raise NoSuchNameError('Selected name is not allowed.')
