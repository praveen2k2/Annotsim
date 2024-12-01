import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import utils

import collections
import copy
import random
import sys
import time
from random import seed
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torchvision import datasets, transforms
from torch import optim
import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from tqdm import tqdm
from helpers import *
from UNet import UNetModel, update_ema_params
from TUVW import UViT, update_ema_params, DINOHead
from UModels.DHUNet import DHUNet
from UModels.CUViT import CUViT
from UModels.UDHVT import UDHVT
from UModels.DiT import DiT_models, DiT_Anomaly, DiT_Anomaly_S_8
torch.cuda.empty_cache()
ROOT_DIR = "./"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='UViT', type=str,
        choices=['UViT', 'UNetModel', 'UDHVT'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=(3, 224, 224), type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.014, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=2000, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=30, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=30, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters and Dataset
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=2, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
        help="""Name of the dataset write Brats if it is Brats2020""")   
    
    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser
def new_args(file):
        # make directories
    for i in ['./model/', "./diffusion-videos/", './diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass
        
    # load the json args
    with open(f'{ROOT_DIR}test_args/{file}.json', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [f'./model/diff-params-ARGS={args["arg_num"]}',
              f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
              f'./diffusion-videos/ARGS={args["arg_num"]}',
              f'./diffusion-training-images/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)
    return args
def train(args_, args):
    """
    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """
    # Convert dictionary to SimpleNamespace
#     args_ = SimpleNamespace(**args)
    
    print("============ init distributed ... ============")
    utils.init_distributed_mode(args_)
    print("============ init random seeds ... ============")
    utils.fix_random_seeds(args_.seed)
    print("============ init get SHA ... ============")
#     print("git:\n  {}\n".format(utils.get_sha()))    
    
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args_)).items())))
    cudnn.benchmark = True  
    print("============ preparing data ... ============")
    transform = DataAugmentationDINO(
        args_.global_crops_scale,
        args_.local_crops_scale,
        args_.local_crops_number,
    )

    dataset = datasets.ImageFolder(args_.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
#     data_loader = init_dataset_loader(dataset, sampler, args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args_.batch_size_per_gpu,
        num_workers=args_.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    max_data = len(data_loader)
    print(f"Data loaded: there are {len(data_loader)} batch images so as {max_data} max_data")    
    
    if args['model_name'] == "UNetModel":
        student = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=3
        )
        teacher = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=3
        )
        
    elif args['model_name'] == "UViT":
        student = UViT(img_size = 224, patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                         depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                         mlp_time_embed=False, num_classes=args["cls_cond"],
                         use_checkpoint=False, conv=True, skip=True)


        teacher = UViT(img_size = 224, patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                         depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                         mlp_time_embed=False, num_classes=args["cls_cond"],
                         use_checkpoint=False, conv=True, skip=True) 
    elif args['model_name'] == "DiT":
        student = DiT_Anomaly(
            input_size=args['img_size'][0],
            in_channels = 3,
            num_classes=args["cls_cond"]
        )
        teacher = DiT_Anomaly(
            input_size=args['img_size'][0],
            in_channels = 3,
            num_classes=args["cls_cond"]
        )
    elif args['model_name'] == "DiT_Anomaly_S_8":
        student = DiT_Anomaly_S_8(
            input_size=args['img_size'][0],
            in_channels = 3,
            num_classes=args["cls_cond"]
        )
        teacher = DiT_Anomaly_S_8(
            input_size=args['img_size'][0],
            in_channels = 3,
            num_classes=args["cls_cond"]
        )        
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    
    start_epoch = 0
    print("***multi-crop wrapper handles forward with inputs of different resolutions***")
    if args['dis']:
        student = utils.MultiCropWrapper(student, DINOHead(in_chans=3, out_chans=3))
        teacher = utils.MultiCropWrapper(teacher, DINOHead(in_chans=3, out_chans=3))
    
    print("***move networks to gpu***")
    student, teacher = student.cuda(), teacher.cuda()
    print("***synchronize batch norms (if any)***")
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args_.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args_.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args['model_name']} network.")    
    
    dino_loss = DINOLoss(
        args_.out_dim,
        args_.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args_.warmup_teacher_temp,
        args_.teacher_temp,
        args_.warmup_teacher_temp_epochs,
        args_.epochs,
    ).cuda()
    
    diffusion = GaussianDiffusionModel(
        args['img_size'],
        betas, 
        loss_weight = args['loss_weight'],
        loss_type = args['loss-type'],
        noise = args["noise_fn"],
        octave = args["octave"],
        frequency = args["frequency"],
        persistence = args["persistence"],
        patch_size = args["patch_size"],
        sigma = args["sigma"],
        img_channels = args["channels"]
    )    
    
    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    # ============ preparing loss ... ============
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args_.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args_.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args_.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    
    fp16_scaler = None
    if args_.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args_.lr * (args_.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args_.min_lr,
        args_.epochs, max_data,
        warmup_epochs=args_.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args_.weight_decay,
        args_.weight_decay_end,
        args_.epochs, max_data,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args_.momentum_teacher, 1,
                                               args_.epochs, max_data)
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args_.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    losses = []
    row_size = min(2, args['Batch_Size'])
#     args['train_save'] = 
    for epoch in range(start_epoch, args_.epochs):
        mean_loss = []
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args_.epochs)
        for it, (images, _) in enumerate(metric_logger.log_every(data_loader, max_data//2, header)):
            it = max_data * epoch + it  # global training iteration
            
            if args["dataset"] == "images":
                if args['dis']:
                    images = [im.cuda(non_blocking=True) for im in images]
                else:
                    images = images[0].cuda()    
                lab = torch.tensor([0], device='cuda:0') if args["cls_cond"] else None
            else:
                x = data["image"]
                if args["cls_cond"] is not None:
                    lab = data["label"]
                    lab = lab.to(device)
                else:
                    lab = args["cls_cond"]
                x = x.to(device)
#             print(len(images))
            loss, estimates, t = diffusion.p_loss(student, images, lab, args)
            
#             teacher_output = teacher((images[:2], t, lab))  # only the 2 global views pass through the teacher
#             dn_loss = dino_loss(estimates[2], teacher_output, epoch)
# #             print(mse)
# #             print(dn_loss)
#             loss = dn_loss+loss
            
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            noisy, est = estimates[1][0], estimates[2][0]
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if args_.clip_grad:
                    param_norms = utils.clip_gradients(student, args_.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args_.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args_.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args_.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args_.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
#                 update_ema_params(ema, model)
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            mean_loss.append(loss.data.cpu())
        if epoch %50==0:
            training_outputs(
                diffusion, images, lab, est, noisy, (epoch, it), row_size, save_imgs=args['save_imgs'],
                save_vids=args['save_vids'], ema=teacher, args=args
            )
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        losses.append(np.mean(mean_loss))
#         if epoch % 200 == 0:
#             time_taken = time.time() - start_time
#             remaining_epochs = args['EPOCHS'] - epoch
#             time_per_epoch = time_taken / (epoch + 1 - start_epoch)
#             hours = remaining_epochs * time_per_epoch / 3600
#             mins = (hours % 1) * 60
#             hours = int(hours)
            
#             vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
#             vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
#             print(
#                     f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
#                     f" {np.mean(vlb):.4f}, "
#                     f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
#                     f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
#                     f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
#                     f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
#                     f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
#                     f"est time remaining: {hours}:{mins:02.0f}\r"
#                     )
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args_,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args_.output_dir, 'checkpoint.pth'))
        if args_.saveckp_freq and epoch % args_.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args_.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args_.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    :param final: bool for final vs checkpoint
    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param ema: ema instance
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    if final:
        torch.save(
                {
                    'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(diffusion, x, lab, est, noisy, Epoch, row_size, ema, args, save_imgs=False, save_vids=False):
    """
    Saves video & images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param noisy: x_t
    :param epoch:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param ema: exponential moving average unet for sampling
    :param save_imgs: bool for saving imgs
    :param save_vids: bool for saving diffusion videos
    :return:
    """
    epoch, it = Epoch
    ema = ema.backbone
    diffusion.train = False
# Check if x is a tuple
    if isinstance(x, list):
        # Reassign x to the first element of the tuple
        x = x[0][:row_size, ...]
    try:
        os.makedirs(f'./diffusion-videos/{args["train_save"]}')
        os.makedirs(f'./diffusion-training-images/{args["train_save"]}')
    except OSError:
        pass
    if save_imgs:
        # for a given t, output x_0, & prediction of x_(t-1), and x_0
        noise = torch.rand_like(x)
        t = torch.randint(990, diffusion.num_timesteps, (x.shape[0],), device=x.device)
        x_t = diffusion.sample_q(x, t, noise)
        temp = diffusion.sample_p(ema, x_t, t, lab)
#         ms = (x - temp["pred_x_0"]).square().cpu()[:row_size, ...]
        
        out = torch.cat(
            (
             x[:row_size, ...].cpu(),
             temp["sample"][:row_size, ...].cpu(),
             temp["pred_x_0"][:row_size, ...].cpu()
            )
        )
        plt.title(f'real, sample, prediction x_0-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')
        plt.savefig(f'./diffusion-training-images/{args["train_save"]}/EPOCH={epoch} it = {it}.png')
        plt.clf()
    else:
        with torch.no_grad():
    #         t = torch.randint(990, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            output = diffusion.forward_backward(
                ema,
                x,
                lab,
                see_whole_sequence=None,
                t_distance=None, denoise_fn='gauss'
            )
            out = torch.cat(
                (
                    x,
                    output.to(x.device)
                )
            )
            plt.title(f'real, prediction x_0-{epoch}epoch')
            plt.rcParams['figure.dpi'] = 150
            plt.grid(False)
            plt.imshow(gridify_output(out, row_size))
            plt.savefig(f'./diffusion-training-images/{args["train_save"]}/EPOCH={epoch}.png')
            plt.clf()        
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0:
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, lab, "half", args['sample_distance'] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, lab, "half", args['sample_distance'] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=50, blit=True,
                    repeat_delay=1000
                    )

            ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.gif')
    plt.close('all')

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.normalize = normalize
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
#         out_dim = 
        self.register_buffer("center", torch.zeros(1, *out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
#         print(student_output.shape)
#         print(teacher_output.shape)
#         print(self.center.shape)
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
#                 print("CE", loss)
#                 print(f"{iq}//{len(teacher_out)} and {v}//{len(student_out)}")
#                 loss = self.VIC(q, student_out[v])
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def VIC(self, x, y):
        
        batch_size, num_features = x.shape
        repr_loss = torch.sum(-x * F.log_softmax(y, dim=-1), dim=-1)
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)
#         print(f"std_loss {std_loss}")
#         print(f"cov_loss {cov_loss}")
        loss = (25.0* repr_loss+ 25.0 * std_loss + 1.0 * cov_loss)/(25.0+25.0+1.0)

        return loss
    
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
if __name__ == '__main__':
    ROOT_DIR = "./"
    args = new_args(300)
#     max_data = 200     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args_ = parser.parse_args()
    print(args_)
    print("Printing dataset names", args_.dataset)
    Path(args_.output_dir).mkdir(parents=True, exist_ok=True)
    train(args_, args)