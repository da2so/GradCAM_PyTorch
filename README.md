# GradCAM

![2](./assets/fig1.png){: .mx-auto.d-block :}

## Requirements

- Pytorch 1.14 
- Python3.6
- CUDA10.1 (optional)


## Running the code

```shell
python main.py --model_path=VGG19 --img_path=examples/catdog --cuda=True
```

Arguments:

- `model_path` - Choose a pretrained model (VGG19, ResNet50, DenseNet169, ...) or saved model (.pt) 
- `img_path` - Image Path
- `cuda` - Use cuda?
