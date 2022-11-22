# https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/helpers.py

from pathlib import Path

from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel, torch


def load_model(model_path, device=None, verbose=False, trace=True, size=640, half=False):
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
    # set logging
    # if not verbose:
    #     LOGGER.setLevel(logging.WARNING)

    # set device if not given
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)

    model = attempt_load(model_path, map_location=device)
    if trace:
        model = TracedModel(model, device, size)

    if half:
        model.half()

    return model


class Yolov7Detector:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = load_model(model_path=self.model_path, device=self.device, trace=True, size=640)
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov7 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = load_model(model_path=self.model_path, device=self.device, trace=True, size=640)

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov7 prediction using loaded model weights.
        Returns results as a yolov7.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results
