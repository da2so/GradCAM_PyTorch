import argparse

from GradCAM import GradCAM

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch GradCAM')

    parser.add_argument('--img_path', type = str, defaul = "examples/elephant.png", help = 'Image Path')

    #Available model list:{'alexnet', 'vgg19', 'resnet50', 'densenet169', 'mobilenet_v2' ,'wide_resnet50_2', ...}
    parser.add_argument('--model_path', type = str, default = "resnet50", help = 'Choose a pretrained model or saved model (.pt)')
    parser.add_argument('--select_t_layer', type = str2bool, default = 'False', help = 'Choose a target layer manually?')

    arg = parser.parse_args()

    gradcam_obj = GradCAM(arg.img_path,arg.model_path,arg.select_t_layer)
    gradcam_obj.build()
