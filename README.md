# GradCAM


![2](./assets/fig1.png)

## Requirements

- Pytorch 1.4.0 
- Python 3.6
- cv2 4.4.0
- matplotlib 3.3.1
- CUDA 10.1 (optional)


## Running the code

```shell
python main.py --model_path=VGG19 --img_path=examples/catdog.png
```

Arguments:

- `model_path` - Choose a pretrained model (VGG19, ResNet50, DenseNet169, ...) or saved model (.pt) 
- `img_path` - Image Path
- `cuda` - Use cuda?
- `cuda_device` - Select a specific GPU device


## Understanding GradCAM

Check my blog!!
[GradCAM in da2so](https://da2so.github.io/2020-08-10-GradCAM/)