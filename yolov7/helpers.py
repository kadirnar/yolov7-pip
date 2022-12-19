# https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/helpers.py

import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from pathlib import Path

from PIL import Image

from yolov7.models.common import autoShape
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel, torch


def load_model(model_path, autoshape=True, device=None, trace=True, size=640, half=False, hf_token=None):
    """
    Creates a specified YOLOv7 model
    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov7 logs will be silent
    Returns:
        pytorch model
    (Adapted from yolov7.hubconf.create)
    """

    # set device if not given
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)
    model = attempt_load(model_path, map_location=device, hf_token=hf_token)
    if trace:
        model = TracedModel(model, device, size)

    if autoshape:
        model = autoShape(model)

    if half:
        model.half()

    return model


if __name__ == "__main__":
    repo_id = "kadirnar/yolov7-v0.1"
    device = "cuda:0"
    model = load_model(repo_id, device, trace=True, size=640, hf_token=None)
    imgs = [Image.open(x) for x in Path("inference/images").glob("*.jpg")]
    results = model(imgs, size=640, augment=False)
