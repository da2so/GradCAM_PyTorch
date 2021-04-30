import numpy as np
import sys 
import cv2

import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from utils import load_image, load_model, cuda_available
from utils import preprocess_image,save , choose_tlayer


class GradCAM():
    def __init__(self,
                img_path, 
                model_path, 
                select_t_layer, 
                class_index = None):

        self.img_path = img_path
        self.model_path = model_path
        self.class_index = class_index
        self.select_t_layer = select_t_layer
        
        # Save outputs of forward and backward hooking
        self.gradients = dict()
        self.activations = dict()
        
        self.model = load_model(self.model_path)
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        #find finalconv layer name
        if self.select_t_layer == False:
            finalconv_after = ['classifier', 'avgpool', 'fc']

            for idx, m in enumerate(self.model._modules.items()):
                if any(x in m for x in finalconv_after): 
                    break
                else:
                    # m[0] is a name of final module before classifer.
                    # m[1] is the final module itself
                    self.finalconv_module = m[1]
            
            # get a last layer of the module
            self.t_layer = self.finalconv_module[-1]
        else:
            # get a target layer from user's input 
            self.t_layer = choose_tlayer(self.model)

        self.t_layer.register_forward_hook(forward_hook)
        self.t_layer.register_backward_hook(backward_hook)
        
    def __call__(self):

        print('\nGradCAM start ... ')

        self.img = load_image(self.img_path)

        #numpy to tensor and normalize
        self.input = preprocess_image(self.img)

        output = self.model(self.input)
        
        if self.class_index == None:
            # get class index of highest prob among result probabilities
            self.class_index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][self.class_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)

        if cuda_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph = True)
        
        gradients = self.gradients['value']
        activations = self.activations['value']
        
        #reshaping
        weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
        weights = weights.reshape(weights.shape[1], 1, 1)
        activationMap = torch.squeeze(activations[0])
           
        #Get gradcam
        gradcam = F.relu((weights*activationMap).sum(0))
        gradcam = cv2.resize(gradcam.data.cpu().numpy(), (224,224))
        save(gradcam, self.img, self.img_path, self.model_path)
        
        print('GradCAM end !!!\n')
