from GradCAM import GradCAM
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')

    parser.add_argument('--img_path', type=str, default="examples/catdog.png", help='Image Path')
    #Pretrained model list:{'AlexNet', 'VGG19', 'ResNet50', 'DenseNet169', 'MobileNet' ,'WideResNet50'}
    parser.add_argument('--model_path', type=str, default="VGG19", help='Chocie a pretrained model or saved model (.pt)')
    #if you use gpu or cpu, set True or False, respectly.
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda or not')
    parser.add_argument('--cuda_device', type=int, default=0, help='Use cuda or not')

    arg = parser.parse_args()

    gradcam=GradCAM(arg.img_path,arg.model_path,arg.cuda_device)
    gradcam()
