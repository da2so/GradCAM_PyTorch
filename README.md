# GradCAM


![2](./assets/fig1.png)

## Requirements

- Pytorch 1.14 
- Python 3.6
- CUDA 10.1 (optional)


## Running the code

```shell
python main.py --model_path=VGG19 --img_path=examples/catdog.png --cuda=True
```

Arguments:

- `model_path` - Choose a pretrained model (VGG19, ResNet50, DenseNet169, ...) or saved model (.pt) 
- `img_path` - Image Path
- `cuda` - Use cuda?



## Understanding GradCAM

Check my blog!!
[GradCAM in da2so](https://da2so.github.io/2020-08-10-GradCAM/)