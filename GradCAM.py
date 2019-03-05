from utils import *

class Gradient():

    def save_gradient(self, grad):
        self.gradient.append(grad)

    def get_gradient(self, class_index):
        feature_output = []
        self.gradient = []
        x = self.input

        for name, module in self.model.features._modules.items():

            x = module(x)
            #find target layer
            if name == self.target_layer:
                #hooking for saving gradients
                x.register_hook(self.save_gradient)
                feature_output += [x]


        # reshaping for classifier step
        x = x.view(x.size(0), -1)

        # classifier step
        output = self.model.classifier(x)

        if class_index == None:
            #index of highest prob among result probabilities
            class_index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][class_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if cuda_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.gradient[-1]


        weights=torch.mean(grads_val,dim=2)
        weights=torch.mean(weights,dim=2)


        return weights

class GradCAM(Gradient):
    def __init__(self,path,model,target_layer):
        self.img_path=path
        self.model=model
        self.target_layer=target_layer

    def activationMap(self):

        self.img=load_image(self.img_path)

        #numpy to tensor
        self.input = preprocess_image(self.img)

        # vgg19
        finalconv_name = 'features'

        # resnet
        # finalconv_name='layer4'

        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output)

        self.model._modules.get(finalconv_name).register_forward_hook(hook_feature)
        self.model(self.input)

        #features_blobs=np.squeeze(features_blobs,axis=0)
        return features_blobs
    def build(self):
        activationMap=self.activationMap()
        weights=self.get_gradient(None)
        #weights=np.reshape(512,1)
        #gradcam=F.relu(np.multiply(activationMap,weights).sum())

        weights=weights.reshape(weights.shape[1],1,1)

        activationMap=torch.squeeze(activationMap[0])

        gradcam=F.relu((weights*activationMap).sum(0))

        gradcam=cv2.resize(gradcam.data.cpu().numpy(),(224,224))


        save(gradcam,self.img,self.img_path)
        print np.shape(gradcam)

