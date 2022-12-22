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
from yolov7.utils.google_utils import attempt_download_from_hub, attempt_download
from yolov7.utils.torch_utils import TracedModel, torch


def load_model(model_path, autoshape=True, device='cpu', trace=True, size=640, half=False, hf_model=False):
    """
    Creates a specified YOLOv7 model
    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        trace (bool): if True, model will be traced
        size (int): size of the input image
        half (bool): if True, model will be in half precision
        hf_model (bool): if True, model will be loaded from huggingface hub    
    Returns:
        pytorch model
    (Adapted from yolov7.hubconf.create)
    """
    if hf_model:
        model_file = attempt_download_from_hub(model_path, hf_token=None)
    else:
        model_file = attempt_download(model_path)
    
    model = attempt_load(model_file, map_location=device)
    if trace:
        model = TracedModel(model, device, size)

    if autoshape:
        model = autoShape(model)

    if half:
        model.half()

    return model


if __name__ == "__main__":
    model_path = "kadirnar/yolov7-v0.1"
    device = "cuda:0"
    model = load_model(model_path, device, trace=True, size=640, hf_model=True)
    imgs = [Image.open(x) for x in Path("inference/images").glob("*.jpg")]
    results = model(imgs, size=640, augment=False)
