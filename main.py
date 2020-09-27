from GradCAM import GradCAM
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')

    parser.add_argument('--img_path', type=str, default="examples/catdog.png", help='Image Path')
    #Pretrained model list:{'AlexNet', 'VGG19', 'ResNet50', 'DenseNet169', 'MobileNet' ,'WideResNet50'}
    parser.add_argument('--model_path', type=str, default="VGG19", help='Choose a pretrained model or saved model (.pt)')
    parser.add_argument('--select_t_layer', type=str2bool, default='True', help='Choose a target layer manually?')
    #if you use gpu or cpu, set True or False, respectly.
    parser.add_argument('--cuda', action='store_true', default='True', help='Use cuda or not')
    parser.add_argument('--cuda_device', type=int, default=0, help='Use cuda or not')


    arg = parser.parse_args()

    gradcam=GradCAM(arg.img_path,arg.model_path,arg.select_t_layer, arg.cuda_device)
    gradcam()
