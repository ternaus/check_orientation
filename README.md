# Check orientation

![](https://habrastorage.org/webt/hb/ru/ii/hbruiiuortx05lxfuokzw1skheu.jpeg)

Models to check if image was rotated by 0, 90, 180, 270 degrees.

## Installation
`pip install -U check_orientation`

### Example inference

Colab notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HIGM_b8rH20N414ROZB0HN9w5p4ko2Bd?usp=sharing)

## Training

### Define the config.
Example at [check_orientation/configs](check_orientation/configs)

### Define the environmental variable `TRAIN_IMAGE_PATH` that points to the folder with train dataset.
Example:
```bash
export TRAIN_IMAGE_PATH=<path to the tranining folder>
```

### Define the environmental variable `VAL_IMAGE_PATH` that points to the folder with validation dataset.
Example:
```bash
export VAL_IMAGE_PATH=<path to the validation folder>
```

### Training
```
python -m check_orientation.train -c <path to config>
```

### Inference

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpu> check_orientation/inference.py \
                                   -i <path to images> \
                                   -c <path to config> \
                                   -w <path to weights> \
                                   -o <output-path> \
                                   --fp16
```

### Pre-trained models
Models were pre-trained on the [OpenImages dataset](https://storage.googleapis.com/openimages/web/index.html).

| Models        | Validation accuracy    | Config file  |
| ------------- |:--------------------:| ------------:|
| resnet18_2020-11-07      | 0.8314 | [Link](check_orientation/configs/2020-11-07.yaml) |
| swsl_resnext50_32x4d|0.8388| [Link](check_orientation/configs/2020-11-08.yaml)|
