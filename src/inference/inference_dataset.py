import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
# from retinaface.pre_trained_models import get_model
from pretrained_model import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector(weights_path=efficient_weight)
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    # face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector = get_model("resnet50_2020-07-20", weights_path=retinaface_weight, max_size=2048,device=device)
    
    face_detector.eval()

    result_file = open(f"{args.dataset}.log", "w")

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    for idx_f, filename in tqdm(enumerate(video_list)):
        
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                
            # print("face_list", face_list)
            # print("idx_list", idx_list)
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

        # write the result in a log file
        gt = target_list[idx_f]
        result_in_str = filename + ", " + f"{pred:.2f}" + ", " + f"{gt:.2f}" + "\n"
        result_file.write(result_in_str)
        result_file.flush()

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')

    result_file.close()


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    # parser.add_argument('-e',dest='efficient_weight',
    #     default='weights/retinaface_resnet50_2020-07-20.pth', type=str)
    # parser.add_argument('-r',dest='retinaface_weight',
    #     default='weights/adv-efficientnet-b4-44fb3a87.pth', type=str)

    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    ROOT = os.path.dirname(os.path.realpath(__file__))
    retinaface_weight = os.path.join(
        ROOT, "../../weights/retinaface_resnet50_2020-07-20.pth")
    efficient_weight = os.path.join(
        ROOT, "../../weights/adv-efficientnet-b4-44fb3a87.pth")


    main(args)

