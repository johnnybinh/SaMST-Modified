import os
project_root = os.path.abspath('../..')
import sys
sys.path.append(project_root)



import random

import yaml
import os
import sys
import numpy as np
import time

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import cv2
from networks.transfer_net import TransformerNet
from loss.vgg import Vgg16
from train_model import utils
import torch.nn.functional as F  # ← missing

## Depth Loss Implementation
## Depth Map Ultilization Function
## Depth Model for Depth loss

model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    midas_transform = midas_transforms.dpt_transform
else:
    midas_transform = midas_transforms.small_transform
for param in midas.parameters():
    param.requires_grad = False
midas = midas.to(device)         
midas.eval()




def calc_depth_map(image: torch.Tensor) -> torch.Tensor:
    """
    Estimate a depth map from an image tensor using MiDaS.

    Args:
        image: Tensor of shape (B, C, H, W), values in [0, 1] or [0, 255].

    Returns:
        Depth map tensor of shape (B, H, W).
    """
    with torch.no_grad():
        if image.dim() == 4:
            image = image[0]                        # (C, H, W)

        img = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

        if img.max() <= 1.0:
            img = (img * 255).astype("uint8")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        prediction = midas(input_batch)

        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.unsqueeze(0)          # (B, H, W)

## Depth Loss Function
def calc_depth_loss(
    depth_map1: torch.Tensor,
    depth_map2: torch.Tensor,
    lambda_grad: float = 0.1,
) -> torch.Tensor:
    """
    Depth loss = L1 pixel difference + weighted gradient consistency.

    Args:
        depth_map1: Depth tensor of shape (B, H, W).
        depth_map2: Depth tensor of shape (B, H, W).
        lambda_grad: Weight for the gradient consistency term.

    Returns:
        Scalar loss tensor.
    """
    assert depth_map1.shape == depth_map2.shape, "Depth maps must have the same shape."

    d1 = depth_map1.unsqueeze(1)   # (B, 1, H, W)
    d2 = depth_map2.unsqueeze(1)

    # Absolute depth difference
    loss_l1 = F.l1_loss(d1, d2)

    # Spatial gradient consistency
    def depth_gradient(x):
        dx = x[:, :, :, :-1] - x[:, :, :, 1:]   # horizontal
        dy = x[:, :, :-1, :] - x[:, :, 1:, :]   # vertical
        return dx, dy

    d1_dx, d1_dy = depth_gradient(d1)
    d2_dx, d2_dy = depth_gradient(d2)
    loss_grad = F.l1_loss(d1_dx, d2_dx) + F.l1_loss(d1_dy, d2_dy)

    return loss_l1 + lambda_grad * loss_grad








def check_paths(opt):
    try:
        if not os.path.exists(opt['save_model_dir']):
            os.makedirs(opt['save_model_dir'])

        if opt['checkpoint_model_dir'] is not None and not (os.path.exists(opt['checkpoint_model_dir'])):
            os.makedirs(opt['checkpoint_model_dir'])
    except OSError as e:
        print(e)
        sys.exit(1)


def train(opt):
    device = torch.device("cuda" if opt['cuda'] else "cpu")

    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    transform = transforms.Compose([
        transforms.ToTensor(),  # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255))  # convert back to [0, 255]
    ])
    train_dataset = datasets.ImageFolder(opt['dataset'], transform)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)  # to provide a batch loader


    style_image = [f for f in os.listdir(opt['style_image'])]
    style_num = len(style_image)
    print('total style number:',style_num)
    print(style_image)

    labels = [i for i in range(0,style_num+1)]
    labels = torch.Tensor(labels).cuda()
    # print(labels)
    # exit(123213)


    transformer = TransformerNet(style_num=style_num)
    print('# MODEL parameters:', sum(param.numel() for param in transformer.parameters()), '\n')
    begin_epoch = 0
    if opt['begin_checkpoint'] is not None:
        state_dict = torch.load(opt['begin_checkpoint'])
        transformer.load_state_dict(state_dict)
        print("load checkpoint model to train")
        begin_epoch = opt['begin_epoch']
    transformer = transformer.to(device)


    optimizer = Adam(transformer.parameters(), opt['lr'])

    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.Resize(opt['style_size']),
        transforms.CenterCrop(opt['style_size']),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])



    content_weight = float(opt['content_weight'])
    style_weight = float(opt['style_weight'])
    ae_weight = float(opt['ae_weight'])
    depth_weight   = float(opt.get('depth_weight', 1e4)) 

    total_epochs = opt['epochs']
    for e in range(begin_epoch + 1,total_epochs+1):

        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_ae_loss = 0.
        agg_depth_loss   = 0.  # Cumulative Loss

        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)

            if n_batch < opt['batch_size']:
                break  # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            optimizer.zero_grad()  # initialize with zero gradients

            batch_style_id = [random.randint(1, style_num) for i in range(count - n_batch, count)]
            style_batch = []
            for i in batch_style_id:
                style = utils.load_image(opt['style_image'] + style_image[i-1], size=opt['style_size'])
                style = style_transform(style)
                style_batch.append(style)

            style = torch.stack(style_batch).to(device)
            features_style = vgg(utils.normalize_batch(style))
            gram_style = [utils.gram_matrix(y) for y in features_style]

            for i in range(n_batch):
                batch_style_id.append(0)

            x = x.repeat(2,1,1,1)
            y,embedding = transformer(x.to(device), style_id=batch_style_id)


            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)


            y = torch.split(y, n_batch , dim=0)  # 按照4这个维度去分，每大块包含2个小块
            y1 = y[0]
            y2 = y[1]

            x = torch.split(x, n_batch , dim=0)
            x1 = x[0]
            x2 = x[1]


            features_y = vgg(y1.to(device))
            features_x = vgg(x1.to(device))

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s)
            style_loss *= style_weight

            ae_loss = ae_weight * mse_loss(y2.to(device),x2.to(device))
            
            #  # ── Depth Loss (output vs content) ────────────────────────────────
            # depth_output  = calc_depth_map(y1)   # stylized
            # depth_content = calc_depth_map(x1)   # content (not style — see note)
 
            # # resize depth maps to match if needed
            # if depth_output.shape != depth_content.shape:
            #     depth_content = F.interpolate(
            #         depth_content.unsqueeze(1),
            #         size=depth_output.shape[-2:],
            #         mode="bicubic",
            #         align_corners=False,
            #     ).squeeze(1)
 
            # depth_loss = depth_weight * calc_depth_loss(depth_output, depth_content)
            # # ─────────────────────────────────────────────────────────────────
            y1_midas = midas(y1.to(device))
            x1_midas = midas(x1.to(device))
            depth_loss = mse_loss(y1_midas,x1_midas)
            depth_loss = depth_weight*depth_loss
            # Hey, Geometric Loss is Missing ?

            total_loss = content_loss + style_loss + ae_loss + depth_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_ae_loss += ae_loss.item()
            agg_depth_loss   += depth_loss.item() 

            if (batch_id + 1) % opt['log_interval'] == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tae: \tdepth: {:.6f} {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                    agg_ae_loss / (batch_id + 1),
                                    agg_depth_loss / (batch_id+1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if opt['checkpoint_model_dir'] is not None and (batch_id + 1) % opt['checkpoint_interval'] == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(opt['checkpoint_model_dir'], ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

        if e % opt['save_interval'] == 0:
            # save model
            transformer.eval().cpu()
            save_model_filename = "epoch_" + str(e) + ".model"
            save_model_path = os.path.join(opt['save_model_dir'], save_model_filename)
            torch.save(transformer.state_dict(), save_model_path)
            print("\ntrained model saved at", save_model_path)
            transformer.to(device).train()


        if e % opt['step_size'] == 0:
            lr = opt['lr'] * (opt['weight_decay'] ** (e // opt['step_size']))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('now learning rate: ',optimizer.state_dict()['param_groups'][0]['lr'])





def main():

    with open('train.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)

    random.seed(7)
    check_paths(opt)
    train(opt)


if __name__ == "__main__":
    main()




