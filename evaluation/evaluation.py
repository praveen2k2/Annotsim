"""
Usage:
    # Fast evaluation (no GIFs, essential metrics only)
    python evaluation/model_evaluation_optimized.py 26
    
    # Full evaluation with GIFs
    python evaluation/model_evaluation_optimized.py 26 --save-gifs
    
    # With mixed precision for extra speed
    python evaluation/model_evaluation_optimized.py 26 --use-amp
    
    # Skip VLB computation (expensive)
    python evaluation/model_evaluation_optimized.py 26 --skip-vlb

    # all options combined
    python evaluation/model_evaluation_optimized.py 26 --save-gifs --skip-vlb --use-amp --eval-batch-size 4
"""

import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve
from matplotlib import animation
from utils.helpers import gridify_output, load_parameters
import utils.dataset as dataset
import numpy as np
from tqdm import tqdm

# Throughput-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- plotting helpers -------------------------
def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    recon = recon.reshape(*real.shape)
    mse1 = ((real - recon).square() * 2) - 1
    mse_threshold1 = mse1 > 0
    mse_threshold1 = (mse_threshold1.float() * 2) - 1
    mse1 = mse1.sum(dim=1, keepdim=True)
    mse_threshold1 = mse_threshold1.sum(dim=1, keepdim=True)

    if save:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        output1 = torch.cat((real, recon))
        output2 = torch.cat((mse1, mse_threshold1))
        ax1.imshow(gridify_output(output1, 2)[..., 0])
        ax2.imshow(gridify_output(output2, 2)[..., 0], cmap="hot")
        ax3.imshow(gridify_output(mask, 1)[..., 0], cmap="hot")
        fig.subplots_adjust(hspace=0.005)
        plt.axis('off')
        plt.savefig(filename)
        plt.close(fig)


def heatmap_d(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    recon = recon.reshape(*real.shape)
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    if save:
        output = torch.cat(
            (real, recon, mse.mean(dim=1, keepdim=True),
             mse_threshold.mean(dim=0, keepdim=True), mask),
            dim=1
        )
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.close(fig)


def heatmap_cls(real: torch.Tensor, recon: torch.Tensor, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = (mse > 0).float() * 2 - 1
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, mse_threshold))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.close(fig)


# ------------------------- evaluation metrics -------------------------
def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=1e-6, mse=None):
    if mse is None:
        mse = (real - recon).square()
        mse = (mse > 0.5).float()
    intersection = torch.sum(mse * real_mask)
    union = torch.sum(mse) + torch.sum(real_mask)
    dice = torch.mean((2. * intersection + smooth) / (union + smooth + 1e-8))
    return dice


def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()


def SSIM(real, recon):
    # real/recon must be HWC ndarrays
    if isinstance(real, np.ndarray) and real.dtype.kind == 'f':
        rmax = float(max(real.max(), recon.max()))
        rmin = float(min(real.min(), recon.min()))
        dr = rmax - rmin
        if dr <= 0:
            dr = 1.0
        return ssim(real, recon, channel_axis=2, data_range=dr)
    else:
        return ssim(real, recon, channel_axis=2)


def IoU(real, recon):
    real = real.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)


def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if isinstance(real_mask, torch.Tensor):
        return roc_curve(real_mask.detach().cpu().numpy().flatten(),
                         square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


class TupleAdapter(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_tuple):
        x, t, lab = input_tuple
        x = x.float()

        if isinstance(t, int):
            t = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        elif isinstance(t, torch.Tensor):
            t = t.to(device=x.device, dtype=torch.long).view(-1)
        else:
            t = torch.tensor(t, dtype=torch.long, device=x.device).view(-1)

        out = self.base(x, timesteps=t, y=lab)
        return out


def _build_model_from_args(args):
    name = args['model_name']
    if name == 'UDHVT':
        from src.models.UModels.UDHVT import UDHVT
        base = UDHVT(
            img_size=args['img_size'][0],
            patch_size=args['patch_size'],
            in_chans=args['channels'],
            embed_dim=args['embed_dim'],
            depth=args.get('depth', 12),
            num_heads=args['num_heads'],
            mlp_ratio=args['mlp_ratio'],
            qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=args['cls_cond'],
            conv=True,
            refinement=args.get('refinement', True),
            skip=True,
            deconvpatch=False,
            use_dec=args.get('mlps', ['MLP', 'MLP', 'MLP']),
            PE_type=args.get('patch_emb', 'SPE'),
        )
    elif name == 'DHUNet':
        from src.models.UModels.DHUNet import DHUNet
        base = DHUNet(
            img_size=args['img_size'][0],
            patch_size=args['patch_size'],
            in_chans=args['channels'],
            embed_dim=args['embed_dim'],
            depth=args.get('depth', 12),
            num_heads=args['num_heads'],
            mlp_ratio=args['mlp_ratio'],
            qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=args['cls_cond'],
            conv=True, skip=True
        )
    else:
        raise NotImplementedError(f"model_name={name} not supported here.")
    return TupleAdapter(base)


def _build_diffusion_from_args(args):
    from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    return GaussianDiffusionModel(
        args['img_size'], betas,
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'],
        noise=args['noise_fn'],
        octave=args.get('octave', 10),
        frequency=args.get('frequency', 128),
        persistence=args.get('persistence', 0.8),
        sigma=args.get('sigma', 4),
        patch_size=args.get('patch_size', 16),
        img_channels=args['channels']
    )


@torch.inference_mode()
def evaluate_anomaly_metrics(test_dataset, diffusion, args, ema, use_amp=False, eval_batch_size=None):
    """Optimized anomaly evaluation with better batching and optional AMP."""
    from torchvision import transforms
    
    os.makedirs("./metrics", exist_ok=True)

    if eval_batch_size is None:
        eval_batch_size = min(args['Batch_Size'], 4)

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Aggregation buffers
    dice_data, ssim_data, iou_vals = [], [], []
    prec_vals, recall_vals, fpr_vals, auc_scores = [], [], [], []

    timestep = min(200, args['T'] - 1)

    for batch in tqdm(loader, desc="Evaluating anomaly metrics"):
        x = batch["image"].to(device, non_blocking=True).float()
        x = x.reshape(-1, args["channels"], *args["img_size"])
        mask = transforms.Resize(tuple(args["img_size"]))(batch["mask"]).to(device, non_blocking=True)

        lab = args.get("cls_cond", None)

        # Forward pass with optional mixed precision
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                recon = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=timestep)
        else:
            recon = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=timestep)

        # Process each sample in the batch
        for idx in range(x.shape[0]):
            x_sample = x[idx:idx+1]
            recon_sample = recon[idx:idx+1]
            mask_sample = mask[idx:idx+1]

            # Binary anomaly map
            sq_err = (x_sample - recon_sample).square().sum(dim=1, keepdim=True)
            pred_mask = (sq_err > 0.5).float()

            # Metrics
            fpr_s, tpr_s, _ = ROC_AUC(mask_sample, pred_mask)
            auc_scores.append(AUC_score(fpr_s, tpr_s))
            dice_data.append(dice_coeff(x_sample, recon_sample, mask_sample, mse=pred_mask).item())

            # SSIM expects HWC numpy
            x_np = x_sample[0].permute(1, 2, 0).detach().cpu().numpy()
            recon_np = recon_sample[0].permute(1, 2, 0).detach().cpu().numpy()
            ssim_data.append(SSIM(x_np, recon_np))

            prec_vals.append(precision(mask_sample, pred_mask).detach().cpu().numpy())
            recall_vals.append(recall(mask_sample, pred_mask).detach().cpu().numpy())
            iou_vals.append(IoU(mask_sample, pred_mask))
            fpr_vals.append(FPR(mask_sample, pred_mask).detach().cpu().numpy())

    # Summary
    print("\n" + "="*60)
    print("Overall Anomaly Detection Metrics:")
    print("="*60)
    print(f"Dice:      {np.mean(dice_data):.4f} ± {np.std(dice_data):.4f}")
    print(f"SSIM:      {np.mean(ssim_data):.4f} ± {np.std(ssim_data):.4f}")
    print(f"IoU:       {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
    print(f"Precision: {np.mean(prec_vals):.4f} ± {np.std(prec_vals):.4f}")
    print(f"Recall:    {np.mean(recall_vals):.4f} ± {np.std(recall_vals):.4f}")
    print(f"FPR:       {np.mean(fpr_vals):.4f} ± {np.std(fpr_vals):.4f}")
    print(f"AUC:       {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print("="*60)

    # Persist summary CSV
    csv_path = f"./metrics/args{args['arg_num']}.csv"
    with open(csv_path, "w") as f:
        f.write("metric,mean,std\n")
        f.write(f"dice,{np.mean(dice_data):.4f},{np.std(dice_data):.4f}\n")
        f.write(f"ssim,{np.mean(ssim_data):.4f},{np.std(ssim_data):.4f}\n")
        f.write(f"iou,{np.mean(iou_vals):.4f},{np.std(iou_vals):.4f}\n")
        f.write(f"precision,{np.mean(prec_vals):.4f},{np.std(prec_vals):.4f}\n")
        f.write(f"recall,{np.mean(recall_vals):.4f},{np.std(recall_vals):.4f}\n")
        f.write(f"fpr,{np.mean(fpr_vals):.4f},{np.std(fpr_vals):.4f}\n")
        f.write(f"auc,{np.mean(auc_scores):.4f},{np.std(auc_scores):.4f}\n")
    print(f"✓ Saved metrics to {csv_path}\n")


@torch.inference_mode()
def testing(testing_dataset_loader, diffusion, args, ema, model, save_gifs=False, skip_vlb=False, use_amp=False):
    """Optimized testing with optional GIF generation and VLB computation."""
    
    if save_gifs:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/', exist_ok=True)

    ema.eval()
    model.eval()
    plt.rcParams['figure.dpi'] = 200

    # GIF generation
    if save_gifs:
        print("\n[Optional] Generating diffusion GIFs...")
        for i in tqdm([*range(100, min(args['sample_distance'], args['T']), 100)], desc="GIF generation"):
            data = next(testing_dataset_loader)
            if args["dataset"] in ("cifar", "carpet"):
                x = data[0].to(device).float()
                lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]
            else:
                x = data["image"].to(device).float()
                lab = args["cls_cond"]

            x = x.reshape(-1, args["channels"], *args["img_size"])
            row_size = min(5, args['Batch_Size'])
            
            fig, ax = plt.subplots()
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence="half", t_distance=i)
            imgs = [[ax.imshow(gridify_output(xx, row_size), animated=True)] for xx in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True, repeat_delay=1000)

            files = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
            ani.save(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.gif')
            plt.close(fig)
        print("✓ GIFs saved\n")
    else:
        print("[Skipped] GIF generation (use --save-gifs to enable)\n")

    # VLB computation
    if not skip_vlb:
        print("Computing VLB statistics...")
        test_iters = 40
        vlb = []
        for _ in tqdm(range(test_iters // max(1, args["Batch_Size"]) + 5), desc="VLB computation"):
            data = next(testing_dataset_loader)
            if args["dataset"] != "cifar":
                x = data["image"].to(device).float()
                lab = args["cls_cond"]
            else:
                x = data[0].to(device).float()
                lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

            x = x.reshape(-1, args["channels"], *args["img_size"])

            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
            else:
                vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
            vlb.append(vlb_terms)

        print(f"Total VLB:  {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb]):.4f} ± "
              f"{np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb]):.4f}")
        print(f"Prior VLB:  {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb]):.4f} ± "
              f"{np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb]):.4f}")
        print(f"VB@t=200:   {np.mean([i['vb'][0][199].cpu().item() for i in vlb]):.4f} ± "
              f"{np.std([i['vb'][0][199].cpu().item() for i in vlb]):.4f}")
        print(f"x0_mse@200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb]):.4f} ± "
              f"{np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb]):.4f}")
        print(f"mse@200:    {np.mean([i['mse'][0][199].cpu().item() for i in vlb]):.4f} ± "
              f"{np.std([i['mse'][0][199].cpu().item() for i in vlb]):.4f}\n")
    else:
        print("[Skipped] VLB computation (use --compute-vlb to enable)\n")

    # PSNR computation (fast, always run)
    print("Computing PSNR...")
    test_iters = 40
    psnr = []
    for _ in tqdm(range(test_iters // max(1, args["Batch_Size"]) + 5), desc="PSNR computation"):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"].to(device).float()
            lab = args["cls_cond"]
        else:
            x = data[0].to(device).float()
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

        x = x.reshape(-1, args["channels"], *args["img_size"])

        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                out = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=args["T"] // 2)
        else:
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=args["T"] // 2)
        psnr.append(PSNR(out, x))

    print(f"Test PSNR: {np.mean(psnr):.4f} ± {np.std(psnr):.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Optimized model evaluation with performance options")
    parser.add_argument('param', nargs='?', default=None, help='Model parameter (e.g., 26, args26, diff-params-ARGS=26)')
    parser.add_argument('--save-gifs', action='store_true', help='Generate diffusion GIFs (slow, creates large files)')
    parser.add_argument('--skip-vlb', action='store_true', help='Skip VLB computation (expensive)')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision for faster inference')
    parser.add_argument('--eval-batch-size', type=int, default=None, help='Batch size for evaluation (default: min(Batch_Size, 4))')
    
    parsed_args = parser.parse_args()

    # Load model parameters
    import sys
    if parsed_args.param:
        sys.argv = [sys.argv[0], parsed_args.param]
    
    args, output = load_parameters(device)
    print(f"\n{'='*60}")
    print(f"Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Args:          {args['arg_num']}")
    print(f"Model:         {args['model_name']}")
    print(f"Noise:         {args['noise_fn']}")
    print(f"Save GIFs:     {parsed_args.save_gifs}")
    print(f"Skip VLB:      {parsed_args.skip_vlb}")
    print(f"Use AMP:       {parsed_args.use_amp}")
    print(f"Device:        {device}")
    print(f"{'='*60}\n")

    # Build models
    ema = _build_model_from_args(args)
    model = _build_model_from_args(args)
    diff = _build_diffusion_from_args(args)

    # Load weights
    ema.base.load_state_dict(output["ema"])
    if "model_state_dict" in output:
        model.base.load_state_dict(output["model_state_dict"])
    else:
        model.base.load_state_dict(output["ema"])

    ema.to(device).eval()
    model.to(device).eval()

    # Data
    _, testing_dataset = dataset.init_datasets("./", args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # 1) Qualitative + VLB/PSNR
    testing(
        testing_dataset_loader, diff, args, ema, model,
        save_gifs=parsed_args.save_gifs,
        skip_vlb=parsed_args.skip_vlb,
        use_amp=parsed_args.use_amp
    )

    # 2) Quantitative anomaly metrics
    try:
        evaluate_anomaly_metrics(
            testing_dataset, diff, args, ema,
            use_amp=parsed_args.use_amp,
            eval_batch_size=parsed_args.eval_batch_size
        )
    except Exception as e:
        print(f"[Warning] Skipping anomaly metrics: {e}")

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
