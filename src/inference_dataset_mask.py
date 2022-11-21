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

    output_list = []
    for idx_vid in tqdm(range(len(video_list))):
        filename = video_list[idx_vid]

        print(f"Processing {filename}")

        # skip real videos for the moment
        # if target_list[idx_vid] == 0:
        #     continue

        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector, image_size=(384, 384))

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float()/255
                output_mask, output_fakenesses = model(img)
                output_mask = output_mask.cpu().detach().numpy()
                output_fakenesses = output_fakenesses.cpu().detach().numpy()

                print("output_mask.shape:", output_mask.shape)
                print("output_fakenesses.shape:", output_fakenesses.shape)
            

            label = "real" if target_list[idx_vid] == 0 else "fake"
            item_folder = os.path.join(output_folder, os.path.basename(filename) + f".{label}")
            os.makedirs(item_folder, exist_ok=True)

            # store the blending mask
            for idx_face in range(len(face_list)):
                face = face_list[idx_face]
                face = np.transpose(face, (1, 2, 0))
                face = np.uint8(face)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
                mask = output_mask[idx_face, 0]
                # print("mask: min,max = ", mask.min(), mask.max())
                
                mask = np.uint8(mask * 255)
                
                cv2.imwrite(os.path.join(item_folder, f"face_{idx_face}.png"), face)
                cv2.imwrite(os.path.join(item_folder, f"mask_{idx_face}.png"), mask)

            # store the prediction of fakeness
            pred_list = []
            idx_img = -1
            print("output_fakenesses: ", output_fakenesses)
            for i in range(len(output_fakenesses)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(output_fakenesses[i])
            pred_res = np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i] = max(pred_list[i])
            # pred = pred_res.mean()
            # print(f"successful prediction: pred={pred}")

        except Exception as e:
            print(e)
            pred_res = np.ones(args.n_frames) * 0.5
            # pred = 0.5
        output_list.append(pred_res)

        print(f"label={target_list[idx_vid]}, pred={pred_res.mean():.2f}")
    
    
    
    target_img_list, output_img_list = extend_framewise_list(target_list, output_list)
    image_auc = roc_auc_score(target_img_list, output_img_list)
    print("target_img_list: ", target_img_list)
    print("output_img_list: ", output_img_list)
    print(f'{args.dataset}| Image-wise AUC: {image_auc:.4f}')

    output_video_list = [preds.mean() for preds in output_list]
    video_auc = roc_auc_score(target_list, output_video_list)
    print("target_video_list: ", target_list)
    print("output_video_list: ", output_video_list)
    print(f'{args.dataset}| Video-wise AUC: {video_auc:.4f}')

    result_file.close()



def extend_framewise_list(target_list, output_list):
    """
    Params
    ------
        target_list: a list of N integers in {0, 1}
        output_list: a list of N arrays, each array stores the frame-wise predictions
    
    Return
    ------
        target_img_list: extend video labels to frame labels
        output_img_list: expanded frame predictions
    """
    assert len(target_list) == len(output_list)

    sizes_of_vectors = [len(v) for v in output_list]

    target_img_list = []
    output_img_list = []
    for i in range(len(target_list)):
        label = target_list[i]
        size = sizes_of_vectors[i]
        target_img_list += [label] * size
        output_img_list += list(output_list[i])

    return target_img_list, output_img_list


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
    parser.add_argument('-n', dest='n_frames', default=64, type=int)
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.realpath(__file__))
    retinaface_weight = os.path.join(
        ROOT, "../weights/retinaface_resnet50_2020-07-20.pth")

    main(args)
