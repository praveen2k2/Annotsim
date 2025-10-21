import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve
from matplotlib import animation
from utils.helpers import gridify_output, load_parameters
import utils.dataset as dataset
import numpy as np

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
        plt.clf()


def heatmap_cls(real: torch.Tensor, recon: torch.Tensor, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = (mse > 0).float() * 2 - 1
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, mse_threshold))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


# ------------------------- metrics -------------------------
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
    # skimage requires data_range for float inputs; infer it robustly from inputs
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

# in evaluation.py
class TupleAdapter(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_tuple):
        x, t, lab = input_tuple

        # Force x to float32 before entering the model
        x = x.float()

        # Ensure timesteps are long tensors on the same device
        if isinstance(t, int):
            t = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        elif isinstance(t, torch.Tensor):
            t = t.to(device=x.device, dtype=torch.long).view(-1)
        else:
            t = torch.tensor(t, dtype=torch.long, device=x.device).view(-1)

        # Base returns a tuple (pred, ..); return it directly so [0] works upstream
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


def evaluate_anomaly_metrics(test_dataset, diffusion, args, ema):
    """Evaluate anomaly detection metrics on a labeled anomalous dataset.

    Computes per-sample and aggregated: Dice, SSIM, IoU, Precision, Recall, FPR, ROC-AUC.
    Saves an aggregated CSV in ./metrics/args{arg_num}.csv
    """
    import os
    from torchvision import transforms
    os.makedirs("./metrics", exist_ok=True)

    # Data loader (no shuffle for reproducibility)
    loader = iter(torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args['Batch_Size'],
        shuffle=False,
        num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    ))

    # Aggregation buffers
    dice_data, ssim_data, iou_vals, prec_vals, recall_vals, fpr_vals, auc_scores = [], [], [], [], [], [], []

    # Fixed timestep used in prior codepaths
    timestep = min(200, args['T'] - 1)

    total = len(test_dataset)
    for i in range(total):
        batch = next(loader)

        # Inputs
        x = batch["image"].to(device).float().reshape(-1, args["channels"], *args["img_size"])
        mask = transforms.Resize(tuple(args["img_size"]))(batch["mask"]).to(device)

        # Labels for class-conditional models (often None)
        lab = args.get("cls_cond", None)

        with torch.no_grad():
            recon = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=timestep)

        # Binary anomaly map via thresholding squared error
        sq_err = (x - recon).square().sum(dim=1, keepdim=True)
        pred_mask = (sq_err > 0.5).float()

        # ROC-AUC over per-pixel scores
        fpr_s, tpr_s, _ = ROC_AUC(mask, pred_mask)
        auc_scores.append(AUC_score(fpr_s, tpr_s))

        # Dice
        dice_data.append(dice_coeff(x, recon, mask, mse=pred_mask).item())

        # SSIM expects HWC numpy
        x_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
        recon_np = recon[0].permute(1, 2, 0).detach().cpu().numpy()
        ssim_data.append(SSIM(x_np, recon_np))

        # Other set metrics
        prec_vals.append(precision(mask, pred_mask).detach().cpu().numpy())
        recall_vals.append(recall(mask, pred_mask).detach().cpu().numpy())
        iou_vals.append(IoU(mask, pred_mask))
        fpr_vals.append(FPR(mask, pred_mask).detach().cpu().numpy())

        if (i + 1) % max(1, (total // 8)) == 0:
            print(
                f"[{i+1}/{total}] Dice {np.mean(dice_data):.3f}, SSIM {np.mean(ssim_data):.3f}, "
                f"Prec {np.mean(prec_vals):.3f}, Rec {np.mean(recall_vals):.3f}"
            )

    # Summary
    print("\nOverall metrics:")
    print(f"Dice: {np.mean(dice_data):.4f} +- {np.std(dice_data):.4f}")
    print(f"SSIM: {np.mean(ssim_data):.4f} +- {np.std(ssim_data):.4f}")
    print(f"IoU: {np.mean(iou_vals):.4f} +- {np.std(iou_vals):.4f}")
    print(f"Precision: {np.mean(prec_vals):.4f} +- {np.std(prec_vals):.4f}")
    print(f"Recall: {np.mean(recall_vals):.4f} +- {np.std(recall_vals):.4f}")
    print(f"FPR: {np.mean(fpr_vals):.4f} +- {np.std(fpr_vals):.4f}")
    print(f"AUC: {np.mean(auc_scores):.4f} +- {np.std(auc_scores):.4f}")

    # Persist summary CSV
    with open(f"./metrics/args{args['arg_num']}.csv", "w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for METRIC in [dice_data, ssim_data, iou_vals, prec_vals, recall_vals, fpr_vals, auc_scores]:
            f.write(f"{np.mean(METRIC):.4f} +- {np.std(METRIC):.4f},")
    print("Saved metrics CSV.")

def testing(testing_dataset_loader, diffusion, args, ema, model):
    import os
    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/', exist_ok=True)

    ema.eval()
    model.eval()
    plt.rcParams['figure.dpi'] = 200

    # sample sequences at a few t's
    for i in [*range(100, min(args['sample_distance'], args['T']), 100)]:
        data = next(testing_dataset_loader)
        if args["dataset"] in ("cifar", "carpet"):
            x = data[0].to(device).float()
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]
        else:
            x = data["image"].to(device).float()
            lab = args["cls_cond"]

        # ensure NCHW for DHUNet (B, C, H, W)
        x = x.reshape(-1, args["channels"], *args["img_size"])

        row_size = min(5, args['Batch_Size'])
        fig, ax = plt.subplots()
        with torch.no_grad():
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence="half", t_distance=i)
        imgs = [[ax.imshow(gridify_output(xx, row_size), animated=True)] for xx in out]
        ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True, repeat_delay=1000)

        files = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
        ani.save(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.gif')
        plt.close(fig)

    # quick VLB snapshots
    test_iters = 40
    vlb = []
    for _ in range(test_iters // max(1, args["Batch_Size"]) + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"].to(device).float()
            lab = args["cls_cond"]
        else:
            x = data[0].to(device).float()
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

        x = x.reshape(-1, args["channels"], *args["img_size"])

        with torch.no_grad():
            vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
        vlb.append(vlb_terms)

    # quick PSNR snapshots
    psnr = []
    for _ in range(test_iters // max(1, args["Batch_Size"]) + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"].to(device).float()
            lab = args["cls_cond"]
        else:
            x = data[0].to(device).float()
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

        x = x.reshape(-1, args["channels"], *args["img_size"])

        with torch.no_grad():
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=args["T"] // 2)
        psnr.append(PSNR(out, x))

    print(
        f"Test set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
        f"{np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
    )
    print(
        f"Test set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
        f"{np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
    )
    print(
        f"Test set vb @ t=200: {np.mean([i['vb'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['vb'][0][199].cpu().item() for i in vlb])}"
    )
    print(
        f"Test set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}"
    )
    print(
        f"Test set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['mse'][0][199].cpu().item() for i in vlb])}"
    )
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")

def main():
    args, output = load_parameters(device)
    print(f"[evaluation] args={args['arg_num']}, model={args['model_name']}, noise={args['noise_fn']}")

    ema = _build_model_from_args(args)
    model = _build_model_from_args(args)
    diff = _build_diffusion_from_args(args)

    # ---- load weights into the underlying base modules ----
    # EMA weights are always present
    ema.base.load_state_dict(output["ema"])

    # Model weights: prefer specific model_state_dict; otherwise fall back to EMA
    if "model_state_dict" in output:
        model.base.load_state_dict(output["model_state_dict"])
    else:
        model.base.load_state_dict(output["ema"])

    # Move to device
    ema.to(device).eval()
    model.to(device).eval()

    # Data
    _, testing_dataset = dataset.init_datasets("./", args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # 1) Qualitative + VLB/PSNR
    testing(testing_dataset_loader, diff, args, ema, model)

    # 2) Quantitative anomaly metrics (if masks are available in dataset)
    try:
        evaluate_anomaly_metrics(testing_dataset, diff, args, ema)
    except Exception as e:
        print(f"[evaluation] Skipping anomaly metrics: {e}")


if __name__ == '__main__':
    main()
