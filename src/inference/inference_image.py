import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
# from pretrained_model import get_model
from preprocess import extract_face
import warnings
import cv2


# warnings.filterwarnings('ignore')
from retinaface.predict_single import Model
from collections import namedtuple
model = namedtuple("model", ["url", "model"])
models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",  # noqa: E501 pylint: disable=C0301
        model=Model,
    )
}
from torch.utils import model_zoo
def get_model(model_name: str,model_dir: str, max_size: int, device: str = "cpu") -> Model:
    model = models[model_name].model(max_size=max_size, device=device)
    state_dict = model_zoo.load_url(models[model_name].url, model_dir=model_dir, progress=True, map_location="cpu")

    model.load_state_dict(state_dict)

    return model

def main(args):
    print("main")
    model=Detector()
    print("model")
    model=model.to(device)
    print("device", device)
    cnn_sd=torch.load(args.weight_name, map_location=device)["model"]
    print("load")
    model.load_state_dict(cnn_sd)
    print("load")
    model.eval()

    print("load_state_dict done")

    frame = cv2.imread(args.input_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=device)
    # face_detector = get_model("resnet50_2020-07-20", args.weight_dir, max_size=max(frame.shape),device=device)
    face_detector.eval()

    face_list=extract_face(frame,face_detector)
    print("face_list", face_list)
    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
        pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    print(f'fakeness: {max(pred):.4f}')


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-i',dest='input_image',type=str)
    # parser.add_argument('-d',dest='weight_dir',type=str)
    args=parser.parse_args()

    print("Hi python")
    print("device", device)
    main(args)

