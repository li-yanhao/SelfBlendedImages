import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
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
# from retinaface.pre_trained_models import get_model
from inference.pretrained_model import get_model
from inference.preprocess import extract_frames
from inference.datasets import init_ffiw, init_ff, init_dfd, init_dfdc, init_dfdcp, init_cdf
import cv2

from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def main(args):

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", weights_path=retinaface_weight, max_size=2048, device=device)

    face_detector.eval()

    result_file = open(f"{args.dataset}.log", "w")

    if args.dataset == 'FFIW':
        video_list, target_list = init_ffiw()
    elif args.dataset == 'FF':
        video_list, target_list = init_ff()
    elif args.dataset == 'DFD':
        video_list, target_list = init_dfd()
    elif args.dataset == 'DFDC':
        video_list, target_list = init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list, target_list = init_dfdcp()
    elif args.dataset == 'CDF':
        video_list, target_list = init_cdf()
    else:
        NotImplementedError

    
    output_folder = "output_mask_blend"
    os.makedirs(output_folder, exist_ok=True)

    for idx_vid in tqdm(range(len(video_list))):
        filename = video_list[idx_vid]

        # skip real videos for the moment
        if target_list[idx_vid] == 0:
            continue

        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector, image_size=(384, 384))

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float()/255
                masks_blend = model(img).cpu().detach().numpy()

                print("masks_blend.shape:", masks_blend.shape)
            

            label = "real" if target_list[idx_vid] == 0 else "fake"
            item_folder = os.path.join(output_folder, os.path.basename(filename) + f".{label}")
            os.makedirs(item_folder, exist_ok=True)

            for idx_face in range(len(face_list)):
                face = face_list[idx_face]
                face = np.transpose(face, (1, 2, 0))
                face = np.uint8(face)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
                mask = masks_blend[idx_face, 0]
                mask = np.uint8(mask * 255)
                
                cv2.imwrite(os.path.join(item_folder, f"face_{idx_face}.png"), face)
                cv2.imwrite(os.path.join(item_folder, f"mask_{idx_face}.png"), mask)


        except Exception as e:
            print(e)
            # pred = 0.5
        # output_list.append(pred)


    result_file.close()


if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_name', type=str)

    parser.add_argument('-d', dest='dataset', type=str)
    parser.add_argument('-n', dest='n_frames', default=32, type=int)
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.realpath(__file__))
    retinaface_weight = os.path.join(
        ROOT, "../weights/retinaface_resnet50_2020-07-20.pth")

    main(args)
