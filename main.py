from GradCAM import *
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')

    parser.add_argument('--img_path', type=str, default="examples/catdog.png", help='Image Path')
    parser.add_argument('--model', type=str, default="vgg19", help='Chocie a Pretrained Model(vgg19,resnet)')
    parser.add_argument('--target_layer', type=str, default="36", help='Target layer of the model')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda or not')
    arg = parser.parse_args()

    model=load_model(arg.model)


    gradcam=GradCAM(arg.img_path,model,arg.target_layer)
    gradcam.build()

