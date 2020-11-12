# check_orientation
Model to check if image was rotated by 90, 180, 270 degrees.

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
