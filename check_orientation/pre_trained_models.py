from collections import namedtuple
from typing import Optional

from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from timm import create_model as timm_create_model
from torch import nn
from torch.utils import model_zoo

model = namedtuple("model", ["url", "model"])

models = {
    "swsl_resnext50_32x4d": model(
        model=timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4),
        url="https://github.com/ternaus/check_orientation/releases/download/v0.0.3/2020-11-16_resnext50_32x4d.zip",
    ),
}


def create_model(model_name: str, activation: Optional[str] = "softmax") -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)

    if activation == "softmax":
        return nn.Sequential(model, nn.Softmax(dim=1))

    return model
