import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp)/len(pred_idx)


def main(args):
    cfg = load_json(args.config)

    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    train_dataset = SBI_Dataset(phase='train', image_size=image_size)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size//2,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=train_dataset.worker_init_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=4,
                                             pin_memory=True,
                                             worker_init_fn=val_dataset.worker_init_fn
                                             )

    model = Detector()

    model = model.to('cuda')

    iter_loss = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    n_epoch = cfg['epoch']
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))

    if args.session == '':
        now = datetime.now()
        args.session = 'sbi_'+now.strftime(os.path.splitext(
            os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")
    else:
        pass
    
    if args.init_weight_name != '':
        model_weight = torch.load(args.init_weight_name)["model"]
        model.load_state_dict(model_weight)

    save_path = 'output/' + args.session + '/'

    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'weights/', exist_ok=True)
    os.makedirs(save_path+'logs/', exist_ok=True)
    logger = log(path=save_path+"logs/", file="losses.logs")

    # criterion = nn.CrossEntropyLoss()
    criterion = F.mse_loss

    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 2
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.
        train_acc = 0.
        model.train(mode=True)
        for step, data in enumerate(tqdm(train_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target_class = data['label'].to(device, non_blocking=True).long()
            mask_blend = data['blend'].to(device, non_blocking=True).float()

            output_mask, output_class = model.training_step(img, mask_blend, target_class)
            loss = F.mse_loss(output, mask_blend) + 

            loss_value = loss.item()
            iter_loss.append(loss_value)
            train_loss += loss_value
            # acc = compute_accuray(F.log_softmax(output, dim=1), target)
            # train_acc += acc
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        # train_accs.append(train_acc/len(train_loader))

        # log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
        #     epoch+1,
        #     n_epoch,
        #     train_loss/len(train_loader),
        #     train_acc/len(train_loader),
        # )
        log_text = "Epoch {}/{} | train loss: {:.4f}, ".format(
            epoch+1,
            n_epoch,
            train_loss/len(train_loader),
        )

        model.train(mode=False)
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        np.random.seed(seed)
        for step, data in enumerate(tqdm(val_loader)):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            mask_blend = data['blend'].to(device, non_blocking=True).float()

            # print("mask_blend.shape:", mask_blend.shape)

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, mask_blend)

            loss_value = loss.item()
            iter_loss.append(loss_value)
            val_loss += loss_value
            # acc = compute_accuray(F.log_softmax(output, dim=1), target)
            # val_acc += acc
            # output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
            # target_dict += target.cpu().data.numpy().tolist()
        val_losses.append(val_loss/len(val_loader))
        # val_accs.append(val_acc/len(val_loader))
        # val_auc = roc_auc_score(target_dict, output_dict)
        # log_text += "val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
        #     val_loss/len(val_loader)
        #     val_acc/len(val_loader),
        #     val_auc
        # )
        val_loss_epoch = val_loss/len(val_loader)

        log_text += "val loss: {:.4f}".format(
            val_loss_epoch
        )


        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(save_path+'weights/', "{}_{:.4f}_loss.tar".format(epoch+1, val_loss_epoch))
            weight_dict[save_model_path] = val_loss_epoch
            torch.save({
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_loss = max([weight_dict[k] for k in weight_dict])

        elif val_loss_epoch <= last_val_loss:
            save_model_path = os.path.join(save_path+'weights/', "{}_{:.4f}_val.tar".format(epoch+1, val_loss_epoch))
            for k in weight_dict:
                if weight_dict[k] == last_val_loss:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_loss_epoch
                    break
            torch.save({
                "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path)
            last_val_loss = max([weight_dict[k] for k in weight_dict])

        logger.info(log_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session', default='')
    parser.add_argument('-i', dest='init_weight_name', default='')
    args = parser.parse_args()
    main(args)
