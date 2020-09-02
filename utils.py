import os
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import functional as F
import torch
from torch.autograd import Variable
from torchvision import models
import torchvision


def load_model(model_name):
    #for saved model (.pt)
    if '.pt' in model_name:
        if torch.typename(torch.load(model_name)) == 'OrderedDict':

            model=Net()
            model.load_state_dict(torch.load(model_name))

        else:
            model=torch.load(model_name)

    #for pretrained model (ImageNet)
    elif model_name=='AlexNet':
        model = models.alexnet(pretrained=True)
    elif model_name=='VGG19':
        model = models.vgg19(pretrained=True)
    elif model_name=='ResNet50':
        model = models.resnet50(pretrained=True)
    elif model_name=='DenseNet169':
        model = models.densenet169(pretrained=True)
    elif model_name=='MobileNet':
        model  = models.mobilenet_v2(pretrained=True)
    elif model_name=='WideResNet50':
        model = models.wide_resnet50_2(pretrained=True)
    
    else:
        print('Choose an available pre-trained model')
        return

    model.eval()
    if cuda_available():
        model.cuda()

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
