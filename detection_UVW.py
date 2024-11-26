import random
import time
import torch.nn as nn
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
# import kornia
import torch.nn.functional as F
from matplotlib import animation
from torchvision import datasets, transforms
# from utils.img2mask import convert_to_masks
import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from TUVW import UViT
from UNetSSCAB import UNetModel, update_ema_params
from UNet import UNetModel, update_ema_params
import cv2
import numpy as np
from tqdm import tqdm 
from MS_SSIM import MS_SSIM_L1_LOSS
from UModels.CUViT import CUViT
from UModels.UDHVT import UDHVT
from UModels.DHUNet import DHUNet
from UModels.DiT import DiT_models, DiT_Anomaly
from torchvision.utils import save_image


import os
from shutil import copyfile


def save_selected_images(image_list, output_dir):
    # Calculate the indices of the 10 images to save
    indices = [int(i * (len(image_list) - 1) / 9) for i in range(10)]
    output_dir = os.path.join(output_dir, 'whole')
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the selected images
    for i in indices:
        tensor = image_list[i]
        print(tensor.shape)
        # Remove batch dimension and save image
        image_path = os.path.join(output_dir, f"image_{i}.png")
        save_image(tensor.squeeze(0), image_path)
        print(f"Saved {image_path} to {image_path}")
    
    print("Finished saving selected images.")


def thesBin(segmentation_mask):
    # Assuming you have a tensor of shape (1, 1, 256, 256) named 'segmentation_mask'
    # You can create a thresholded binary mask by setting a threshold value
    threshold_value = 0.5  # Adjust this value based on your requirement
    
    # Apply thresholding to create a binary mask
    binary_mask = (segmentation_mask > threshold_value).float()
    
    # If you want to convert it back to integer (0 or 1)
    binary_mask_int = binary_mask.int()
    
    # If you want to keep the original tensor and update it with thresholding:
    segmentation_mask_thresholded = torch.threshold(segmentation_mask, threshold_value, 0)
    return segmentation_mask_thresholded


def MorphOperat(binary_mask):
    # Assuming you have a tensor of shape (1, 1, 256, 256) named 'binary_mask'
    # You can perform dilation and erosion as follows:
    
    # Convert the binary mask to uint8 for kornia operations
    binary_mask = torch.clamp(torch.round(binary_mask * 255.0), 0, 255).byte().cpu()

    # Define a structuring element for dilation and erosion
    kernel_size = 3
    structuring_element = torch.ones(1, 1, kernel_size, kernel_size)

    # Perform dilation
    dilated_mask = F.conv2d(binary_mask.float(), structuring_element.float(), padding=1) > 0

    # Perform erosion
    eroded_mask = 1 - F.conv2d((1 - binary_mask).float(), structuring_element.float(), padding=1) > 0
    
    return dilated_mask, eroded_mask

def batch_threshold(batch_images, threshold_value):
    """
    Apply thresholding to a batch of grayscale images.

    Args:
        batch_images (numpy.ndarray): A 4D numpy array representing the batch of grayscale images.
                                     The shape should be (batch_size, 1, height, width).
        threshold_value (int): The threshold value used for thresholding.

    Returns:
        numpy.ndarray: A 4D numpy array representing the batch of thresholded binary masks.
                       The shape is the same as the input batch_images.
    """
    batch_images = torch.clamp(torch.round(batch_images * 255.0), 0, 255).byte().cpu().numpy()
    batch_size, _, height, width = batch_images.shape
    batch_thresholded = np.empty_like(batch_images)
    batch_Adthresholded = np.empty_like(batch_images)
   
    for i in range(batch_size):
        image = batch_images[i, 0]  # Extract the grayscale image from the batch
        _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        # Apply adaptive thresholding
        binary_adpmask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        batch_thresholded[i, 0] = binary_mask
        batch_Adthresholded[i, 0] = binary_adpmask
    return torch.from_numpy(batch_thresholded),  torch.from_numpy(batch_Adthresholded)


def calculate_ece(predicted_probabilities, ground_truth_segmentation, num_bins=10):
    # Define the bin limits
    bin_limits = np.linspace(0, 1, num_bins + 1)

    # Initialize variables to track ECE components
    ece = 0.0
    total_samples = 0

    # Calculate ECE
    for bin_idx in range(num_bins):
        bin_lower, bin_upper = bin_limits[bin_idx], bin_limits[bin_idx + 1]

        # Mask for pixels falling in the current probability bin
        bin_mask = (predicted_probabilities >= bin_lower) & (predicted_probabilities < bin_upper)

        # Calculate the number of pixels in this bin
        bin_samples = np.sum(bin_mask)

        if bin_samples > 0:
            # Calculate the proportion of correctly predicted pixels in this bin
            bin_accuracy = np.mean(ground_truth_segmentation[bin_mask])

            # Calculate the average confidence in this bin
            bin_confidence = np.mean(predicted_probabilities[bin_mask])

            # Update the ECE with the contribution from this bin
            ece += bin_samples * np.abs(bin_accuracy - bin_confidence)
            total_samples += bin_samples

    if total_samples > 0:
        # Normalize ECE
        ece /= total_samples

    return ece


def connected_component_analysis_batch(binary_masks, rgb = False):
    """
    Perform connected component analysis on a batch of binary masks using OpenCV.

    Args:
        binary_masks (numpy.ndarray): A 3D numpy array representing the batch of binary masks.
                                     The shape should be (batch_size, height, width).

    Returns:
        list: A list of NumPy arrays, where each array represents the connected components for
              the corresponding image in the batch. The output is a list of 2D arrays, where each
              array is of the same size as the input image and contains the connected component
              labels for each pixel.
    """
    binary_masks_uint8 = (binary_masks * 255).to(torch.uint8)

   
    batch_connected_components = []
    boinary_mask = []
    for i in range(binary_masks.shape[0]):
        binary_masks_uint8 = (binary_masks[i] * 255).to(torch.uint8)
        binary_masks_np = binary_masks_uint8.squeeze().cpu().numpy()
        if binary_masks_np.shape[0] == 4:
            binary_masks_np = binary_masks_np.transpose(1, 2, 0)  
            # Convert RGBA to grayscale
            binary_masks_np = cv2.cvtColor(binary_masks_np, cv2.COLOR_RGBA2GRAY)
        elif binary_masks_np.shape[0] == 3:
            binary_masks_np = binary_masks_np.transpose(1, 2, 0)  
            # Convert BGR to grayscale
            binary_masks_np = cv2.cvtColor(binary_masks_np, cv2.COLOR_BGR2GRAY)
            
        threshold_value, thresholded_np = cv2.threshold(binary_masks_np, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         thresholded_tensor = torch.tensor(thresholded_np, dtype=torch.uint8).unsqueeze(0).unsqueeze(0) 
        
        boinary_mask.append(thresholded_np)

        # Perform connected component analysis
        num_labels, labels = cv2.connectedComponents(thresholded_np)

         # Find the label with the largest area (excluding background label 0)
        largest_label = np.argmax(np.bincount(labels.ravel())[1:]) + 1
        
        # Create the largest connected component mask
        largest_connected_component = (labels == largest_label).astype(np.uint8)
        
        
        # Append the connected components to the batch list
        batch_connected_components.append(largest_connected_component)
    batch_connected_components = np.array(batch_connected_components)
    boinary_mask = np.array(boinary_mask)
    return torch.from_numpy(boinary_mask).unsqueeze(0), torch.from_numpy(batch_connected_components).unsqueeze(0)


def anomalous_validation_1():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    args, output = load_parameters(device)
#     args.update({"patch_size": 16})
#     args.update({ "embed_dim": 256})    
#     args.update({"refinement":False})
#     args.update({"Batch_Size": 1})
# # #     args.update({"model_name": "UViT"})
#     args.update({"mlps":["MLP", "MLP", "MLP"]})
# #     print(args["Batch_Size"])
    print(f"args{args}")
    print(args['channels'])
    if args['model_name'] == "CUViT":
            model = CUViT(img_size = args['img_size'][0], patch_size=4, in_chans=args["channels"], embed_dim = args['base_channels'],
                          depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                          mlp_time_embed=False, num_classes=args["cls_cond"],
                          use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UViT":
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                     depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=args["cls_cond"],
                     use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UDHVT":
        model = UDHVT(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=12, 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      use_checkpoint=False,
                      num_classes=args["cls_cond"],
                      conv=True, refinement=args["refinement"], skip=True, deconvpatch = False, use_dec = args["mlps"],
                      PE_type = args["patch_emb"])
    elif args['model_name'] == "DHUNet":
        model = DHUNet(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=args['depth'], 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
    elif args['model_name'] == "DiT":
        model = DiT_Anomaly(
            input_size=args['img_size'][0],
            num_classes=args["cls_cond"]
        )
    elif args['model_name'] == "UNetModel":
        model = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=args["channels"]
                )
    else:
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = 128,
                     depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=None,
                     use_checkpoint=False, conv=True, skip=True, SSC = True)
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    
    diff = GaussianDiffusionModel(
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
    diff.train = False
    model.load_state_dict(output["ema"])
    model.to(device)
    model.eval()
    ROOT_DIR = "./"
#     print(args)
    _, ano_dataset = dataset.init_datasets(ROOT_DIR, args)

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 600

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
#     for i in ano_dataset.slices.keys():
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass
    t_range = np.linspace(50, 400, 2).astype(np.int32)
    dice_data = []
    start_time = time.time()
    for i in range(200):
#     for i in range(len(ano_dataset)):
        new = next(loader)
        if new['filenames']==['Y7.png']:
            img = new["image"].to(device)
            img = img.reshape(args["Batch_Size"], args["channels"], *args["img_size"])
            print(new["mask"].shape)
    #         img_mask = new["mask"].reshape(args["Batch_Size"], -1, *args["img_size"])
            if args["cls_cond"] is not None:
                lab = new["label"]
                lab = lab.to(device)
            else:
                lab = args["cls_cond"]
            lab = torch.tensor([0], device='cuda:0')
            img_mask = transforms.Resize(args['img_size'], antialias=True)(new["mask"])
            img_mask = img_mask.reshape(args["Batch_Size"], -1, *args["img_size"])
            print(img_mask.shape)
    #         img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
            img_mask = img_mask.to(device)            
            for timestep in t_range:
                if timestep == t_range[-1]:
                    see_whole_sequence = 'whole'
                else:
                    see_whole_sequence = None
                samples = diff.forward_backward(
                    model,
                    img.reshape(args["Batch_Size"], args["channels"], *args["img_size"]),
                    lab,
                    see_whole_sequence=see_whole_sequence,
                    t_distance=timestep, denoise_fn = args["noise_fn"]
                )

                output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
                              f'{new["filenames"][0]}' \
                              f't={timestep}_v={args["frequency"]}'
                output_directory = os.path.dirname(output_name)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)  
                if see_whole_sequence=='whole':
                    output = samples[-1].to(device)
                else: 
                    output = samples
                mse = (img - output).square()
                print("outputs", len(samples))
                ano_mask2, ano_mask4 = connected_component_analysis_batch(mse)
                ano_mask3 = MorphOperat(ano_mask2)
                
                outputs = torch.cat(
                    (
                        img, 
                        output.to(device),
                        img_mask,
                        mse.to(device), 
                        torch.cat((ano_mask2.to(device), ) * args["channels"], dim=1),
                        torch.cat((ano_mask4.to(device), ) * args["channels"], dim=1),
                        torch.cat((ano_mask3[1].to(device), ) * args["channels"], dim=1)                        
                    )
                )   
                print(outputs.shape)
                plt.imshow(gridify_output(outputs, 7)[..., 0], cmap = 'gray')
                plt.axis('off')
                plt.savefig(output_name + ".png")
                plt.close('all')
                time_taken = time.time() - start_time
                remaining_epochs = 22 - i
                time_per_epoch = time_taken / (i + 1)
                hours = remaining_epochs * time_per_epoch / 3600
                mins = (hours % 1) * 60
                hours = int(hours)

                print(
                        f"file: {new['filenames'][0]}, "
                        f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                        f"remaining time: {hours}:{mins:02.0f}"
                        )
                if see_whole_sequence == 'whole':
                    save_selected_images(samples, output_dir = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}')                
            break


def anomalous_metric_calculation():
    """
    Iterates over 4 anomalous slices for each Volume, returning diffused video for it,
    the heatmap of that & detection method (A&B) or C
    :return:
    """
    
    
    ROOT_DIR = "./"
    args, output = load_parameters(device)
    in_channels = args["channels"]
#     args.update({"cls_cond": None})
    args.update({"Batch_Size": 1})
#     args.update({"refinement":True})
# # #     args.update({"model_name": "UViT"})
#     args.update({"mlps":["DAFF", "MLP", "DDAFF"]})
    if args["dataset"].lower() == "leather":
        in_channels = 3
#     args.update({"model_name": "UNetModel"})
    print(f"args{args['arg_num']}")
    if args['model_name'] == "CUViT":
            model = CUViT(img_size = args['img_size'][0], patch_size=4, in_chans=args["channels"], embed_dim = args['base_channels'],
                          depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                          mlp_time_embed=False, num_classes=args["cls_cond"],
                          use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UViT":
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                     depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=args["cls_cond"],
                     use_checkpoint=False, conv=True, skip=True)
#     elif args['model_name'] == "UDHVT":
    elif args['model_name'] == "UDHVT":
        model = UDHVT(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=12, 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      use_checkpoint=False,
                      num_classes=args["cls_cond"],
                      conv=True, refinement=args["refinement"], skip=True, deconvpatch = False, use_dec = args["mlps"])
    elif args['model_name'] == "DHUNet":
        model = DHUNet(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=args['depth'], 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
    elif args['model_name'] == "UNetModel":
        model = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=args["channels"]
                )
    elif args['model_name'] == "DiT":
        model = DiT_Anomaly(
            input_size=args['img_size'][0],
            num_classes=args["cls_cond"]
        )
    else:
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = 128,
                     depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=None,
                     use_checkpoint=False, conv=True, skip=True, SSC = True)
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )

    model.load_state_dict(output["ema"])
    model.to(device)
    model.eval()
    _, d_set = dataset.init_datasets(ROOT_DIR, args)
    d_set_size = len(d_set)
#         * 4
#     loader = iter(torch.utils.data.DataLoader(d_set, batch_size=1, shuffle=False))
#             )
    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 200

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    AUC_scores = []

    start_time = time.time()
    ECE = []
    for i in range(d_set_size):
        new = next(loader)
        image = new["image"].to(device)
        image = image.reshape(1, args["channels"], *args["img_size"])
        mask = new["mask"].to(device)
        mask = transforms.Resize(args['img_size'], antialias=True)(mask)

        if args["cls_cond"] is not None:
            lab = new["label"]
            lab = lab.to(device)
        else:
            lab = args["cls_cond"]
#         timestep = 60
        timestep = random.randint(100, 150)
#         print(lab)
#         lab = torch.tensor([0], device='cuda:0')
        output = diff.forward_backward(
            model,
            image.reshape(args["Batch_Size"], args["channels"], *args["img_size"]),
            lab,
            see_whole_sequence=None,
            t_distance=timestep, denoise_fn=args["noise_fn"]
        )
        
#         mse = (image - output).square()
        mse = (image - output).abs()
#             mse = (mse > 0.5).float()
#             mse = mse.sum(dim=1)
        ano_mask1, ano_mask2 = connected_component_analysis_batch(mse)
#         print(ano_mask1.shape)
        mse = ano_mask1.to(device)
#         mse = torch.cat((mse.to(device), ) * args["channels"], dim=1)
#         print(mse.shape)
#         if mse.is_contiguous():
#             print("The mse is contiguous.")
#         if mask.is_contiguous():
#             print("The mask is contiguous.") 
#         print(mask)
        fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC((mask > 0.5).float(), (mse).float())
        
        ece = calculate_ece(mse.cpu().numpy(), mask.cpu().numpy())
        ECE.append(ece)
        print("Expected Calibration Error (ECE) for a single image:", ece)

        AUC_scores.append(evaluation.AUC_score(fpr_simplex, tpr_simplex))
#         mse = (mse > 0.5).float()
        # print(img.shape, output.shape, img_mask.shape, mse.shape)
        dice_data.append(
                evaluation.dice_coeff(
                        image, output.to(device),
                        mask, mse=mse
                        ).item()
                )
#         print(image.shape)
#         print(output.shape)
        ssim_data.append(
                evaluation.SSIM(
                        image.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1]),
                        output.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1])
                        )
                )
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
#         plt.axis('off')
#         try:
#             os.makedirs(
#                     f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}'
#                     )
#         except OSError:
#             pass
# #         imgs = [[ax.imshow(gridify_output(x, 5), animated=True)] for x in output]
# #         ani = animation.ArtistAnimation(
# #                 fig, imgs, interval=50, blit=True,
# #                 repeat_delay=1000
# #                 )
#         temp = os.listdir(
#                 f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}'
#                 )

#         output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/' \
#                       f'{new["filenames"][0]}' \
#                       f't={timestep}'
# #         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
#         output1 = torch.cat((image, output.to(device)))
# #         output1 = output1.reshape(8, 1, real.shape[-2], real.shape[-1])
# #         print("output1", output1.shape)
#         output2 = torch.cat((mse, mask))
# #         output3 = torch.cat((mse2, mse_threshold2, mask))
#         ax1.imshow(gridify_output(output1, 2)[..., 0])
# #         ax1.set_title('Image and Reconstrction')
#         ax2.imshow(gridify_output(output2, 2)[..., 0])

#         fig.subplots_adjust(hspace=0.005)
#         plt.axis('off')
#         plt.savefig(output_name + ".png")
#         plt.close('all')
        precision.append(evaluation.precision(mask, mse).cpu().numpy())
        recall.append(evaluation.recall(mask, mse).cpu().numpy())
        IOU.append(evaluation.IoU(mask, mse))
        FPR.append(evaluation.FPR(mask, mse).cpu().numpy())
        print(f'{i} out of {d_set_size} done')
#         plt.close('all')

        if i % 8 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = d_set_size - i
            time_per_epoch = time_taken / (i + 1)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            print(
                    f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                    f"remaining time: {hours}:{mins:02.0f}"
                    )

        if i % 4 == 0: # and (args["dataset"].lower() != "carpet" and args["dataset"].lower() != "leather"):
            print(f"file: {new['filenames'][0][-9:-4]}")
            print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
            print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
            print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
            print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
            print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
            print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")
            print(f"AUC: {np.mean(AUC_scores[-4:])} +- {np.std(AUC_scores[-4:])}")
            print(f"ECE: {np.mean(ECE[-4:])} +- {np.std(ECE[-4:])}")            
            print("\n")

    print()
    print("Overall: ")
    print(f"Dice coefficient: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data)} +- {np.std(ssim_data)}")
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")
    print(f"AUC: {np.mean(AUC_scores)} +- {np.std(AUC_scores)}")
    print(f"ECE: {np.mean(ECE)} +- {np.std(ECE)}")    
    with open(f"./metrics/args{args['arg_num']}.csv", mode="w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for METRIC in [dice_data, ssim_data, IOU, precision, recall, FPR, AUC_scores]:
            f.write(f"{np.mean(METRIC):.4f} +- {np.std(METRIC):.4f},")


def graph_data():
    ROOT_DIR = "./"
    args, output = load_parameters(device)
    in_channels = args["channels"]
    args.update({"Batch_Size": 1})
#     args.update({"model_name": "UNetModel"})
    args.update({"cls_cond": None})
    print(f"args{args}")
    
    if args['model_name'] == "CUViT":
            model = CUViT(img_size = args['img_size'][0], patch_size=4, in_chans=args["channels"], embed_dim = args['base_channels'],
                          depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                          mlp_time_embed=False, num_classes=args["cls_cond"],
                          use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UViT":
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                     depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=args["cls_cond"],
                     use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UDHVT":
        model = UDHVT(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=12, 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, refinement=args["refinement"], skip=True, deconvpatch = False, use_dec = args["mlps"])
    elif args['model_name'] == "DHUNet":
        model = DHUNet(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=args['depth'], 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
    elif args['model_name'] == "UNetModel":
        model = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=args["channels"]
                )
    else:
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = 128,
                     depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=None,
                     use_checkpoint=False, conv=True, skip=True, SSC = True)
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diff = GaussianDiffusionModel(
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

    model.load_state_dict(output["ema"])
    model.to(device)
    model.eval()
    training_dataset, ano_dataset = dataset.init_datasets(ROOT_DIR, args)
#     ano_dataset = dataset.AnoBratsDataset(ROOT_DIR)
#     ano_dataset = dataset.HnABRATS(ROOT_DIR, is_train=False)
#     ano_dataset = dataset.MatTDataset(di_r)
#     ano_dataset = dataset.BRATSDataset("./data/brats/testing", test_flag=True)
#     ano_dataset = dataset.AnomalousMRIDataset(
#             ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
#             slice_selection="iterateKnown_restricted", resized=False
#             )
    loader = dataset.init_dataset_loader(ano_dataset, args, shuffle=False)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./metrics/')
    except OSError:
        pass

    try:
        os.makedirs(f'./metrics/ARGS={args["arg_num"]}')
    except OSError:
        pass
    t_range = np.linspace(50, 800, 50).astype(np.int32)
    # t_range = np.linspace(0, 999, 1).astype(np.int32)

    start_time = time.time()
    files_to_complete = defaultdict(str, {"19691": False, "18756": False})
    for i in range(1):

        dice_data = []
        ssim_data = []
        precision = []
        recall = []
        IOU = []
        FPR = []
        new = next(loader)
        img = new["image"].to(device)
        img = img.reshape(args["Batch_Size"], args["channels"], *args["img_size"])
        img_mask = new["mask"].to(device)
        if args["cls_cond"] is not None:
            lab = data["label"]
            lab = lab.to(device)
        else:
            lab = args["cls_cond"]
        img_mask = transforms.Resize(args["img_size"], antialias=True)(img_mask)
        for t in t_range:
            output = diff.forward_backward(
                model,
                img.reshape(args["Batch_Size"], args["channels"], *args["img_size"]),
                lab,
                see_whole_sequence=None,
                t_distance=t, denoise_fn=args["noise_fn"]
            )
#             img = img[0]
#             output = output[0]
            
            mse = (img - output).square()
#             mse = (mse > 0.5).float()
#             mse = mse.sum(dim=1)
            ano_mask1, ano_mask2 = connected_component_analysis_batch(mse)
            ano_mask3 = MorphOperat(ano_mask2)

            mse= ano_mask1.to(device)
            img_mask, _ = connected_component_analysis_batch(img_mask)
            img_mask = img_mask.to(device)
            dice_data.append(evaluation.dice_coeff(img, output.to(device), img_mask, mse.to(device)).item())
            print("step>>>", t)
            ssim_data.append(
                    evaluation.SSIM(
                            img.reshape(*args["img_size"], args["channels"]),
                            output.reshape(*args["img_size"], args["channels"])
                            )
                    )
#             print(ssim_data)
            precision.append(evaluation.precision(img_mask, mse).cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).cpu().numpy())
            output_name = f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0]}.png'
            output_directory = os.path.dirname(output_name)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            if t%5==0 or t%1==0 or t==0:
                print(t, dice_data[-1], ssim_data[-1], precision[-1], recall[-1], IOU[-1])

                plt.plot(t_range[:len(dice_data)], dice_data, label="dice")
                plt.plot(t_range[:len(dice_data)], IOU, label="IOU")
                plt.plot(t_range[:len(dice_data)], precision, label="precision")
                plt.plot(t_range[:len(dice_data)], recall, label="recall")
                plt.legend(loc="upper right")
                ax = plt.gca()
                ax.set_ylim([0, 1])
                plt.savefig(output_name)
                plt.clf()
#         files_to_complete[new['filenames'][0][-9:-4]] = True
        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames']}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
        print(
                f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
                f" {np.std(ssim_data)}"
                )
        print(f"Dice: {np.mean(dice_data)} +- {np.std(dice_data)}")
        print(
                f"Structural Similarity Index (SSIM): {np.mean(ssim_data)} +-"
                f" {np.std(ssim_data)}"
                )
        print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
        print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
        print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")

        plt.plot(t_range, dice_data, label="dice")
        plt.plot(t_range, IOU, label="IOU")
        plt.plot(t_range, precision, label="precision")
        plt.plot(t_range, recall, label="recall")
        plt.legend(loc="upper right")
        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.savefig(output_name)
        plt.clf()
        with open(f'{output_directory}/Output.csv', mode="w") as f:

            f.write(",".join(["timestep", "Dice", "SSIM", "IOU", "Precision", "Recall", "FPR"]))
            f.write("\n")
            for i in range(1000):
                f.write(
                        f"{i:04}," + ",".join(
                                [f"{j:.4f}" for j in [dice_data[i], ssim_data[i], IOU[i], precision[i],
                                                      recall[i], FPR[i]]]
                                )
                        )
                f.write("\n")

                
def model_selection(args):
    if args['model_name'] == "CUViT":
        model = CUViT(img_size = args['img_size'][0], patch_size=4, in_chans=args["channels"], embed_dim = args['base_channels'],
                      depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                      mlp_time_embed=False, num_classes=args["cls_cond"],
                      use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UViT":
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                     depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=args["cls_cond"],
                     use_checkpoint=False, conv=True, skip=True)
    elif args['model_name'] == "UDHVT":
        model = UDHVT(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=12, 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
    elif args['model_name'] == "DHUNet":
        model = DHUNet(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=args['depth'], 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
    elif args['model_name'] == "UNetModel":
        model = UNetModel(
                args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=args["channels"]
                )
    else:
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = 128,
                     depth=12, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=None,
                     use_checkpoint=False, conv=True, skip=True, SSC = True)
    return model
def diffusion_selection(args):
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diff = GaussianDiffusionModel(
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

    return diff
def roc_data():
    sys.argv[1] = "64"
    args_simplex, output_simplex = load_parameters(device)
    sys.argv[1] = "73"
    args_hybrid, output_hybrid = load_parameters(device)
    sys.argv[1] = "40"
    args_gauss, output_gauss = load_parameters(device)

    unet_simplex = model_selection(args_simplex)
    
    unet_hybrid = model_selection(args_hybrid)
    
    unet_gauss = model_selection(args_gauss)
    
    diff_simplex = diffusion_selection(args_simplex)
    diff_gauss = diffusion_selection(args_gauss)

    unet_hybrid.load_state_dict(output_hybrid["ema"])
    unet_simplex.load_state_dict(output_simplex["ema"])
    unet_gauss.load_state_dict(output_gauss["ema"])
    unet_simplex.eval()
    unet_gauss.eval()

    import Comparative_models.CE as CE
    sys.argv[1] = "103"
    args_GAN, output_GAN = load_parameters(device)
    args_GAN["Batch_Size"] = 1
    print(args_GAN)
    netG = CE.Generator(
            start_size=args_GAN['img_size'][0], out_size=args_GAN['inpaint_size'], dropout=args_GAN["dropout"]
            )

    netG.load_state_dict(output_GAN["generator_state_dict"])
    netG.eval()
    ano_dataset_128 = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args_GAN['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader_128 = dataset.init_dataset_loader(ano_dataset_128, args_GAN, False)

    overlapSize = args_GAN['overlap']
    input_cropped = torch.FloatTensor(args_GAN['Batch_Size'], 1, 128, 128)

    ano_dataset_256 = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args_simplex['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )
    loader_256 = dataset.init_dataset_loader(ano_dataset_256, args_simplex, False)
    plt.rcParams['figure.dpi'] = 200

    try:
        os.makedirs(f'./metrics/')
    except OSError:
        pass

    try:
        os.makedirs(f'./metrics/ROC_data_3/')
    except OSError:
        pass
    t_distance = 250

    simplex_sqe = []
    gauss_sqe = []
    GAN_sqe = []
    hybrid_sqe = []
    img_128 = []
    img_256 = []
    simplex_AUC = []
    gauss_AUC = []
    GAN_AUC = []
    hybrid_AUC = []
    for i in range(len(ano_dataset_256)):

        new_256 = next(loader_256)
        img_256_whole = new_256["image"].to(device)
        img_256_whole = img_256_whole.reshape(img_256_whole.shape[1], 1, *args_simplex["img_size"])
        img_mask_256_whole = dataset.load_image_mask(
                new_256['filenames'][0][-9:-4], args_simplex['img_size'],
                ano_dataset_256
                )
        img_mask_256_whole = img_mask_256_whole.to(device)
        img_mask_256_whole = (img_mask_256_whole > 0).float()

        new_128 = next(loader_128)
        img_128_whole = new_128["image"].to(device)
        img_128_whole = img_128_whole.reshape(img_128_whole.shape[1], 1, *args_GAN["img_size"])
        img_mask_128_whole = dataset.load_image_mask(
                new_128['filenames'][0][-9:-4], args_GAN['img_size'],
                ano_dataset_128
                )

        for slice_number in range(4):
            img = img_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_mask = img_mask_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_256.append(img_mask.detach().cpu().numpy().flatten())
            # for slice_number in range(4):
            unet_simplex.to(device)

            output_simplex = diff_simplex.forward_backward(
                    unet_simplex, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_simplex["noise_fn"]
                    )

            unet_simplex.cpu()

            mse_simplex = (img - output_simplex).square()
            simplex_sqe.append(mse_simplex.detach().cpu().numpy().flatten())

            fpr_simplex, tpr_simplex, threshold_simplex = evaluation.ROC_AUC(img_mask, mse_simplex)
            simplex_AUC.append(evaluation.AUC_score(fpr_simplex, tpr_simplex))

            unet_hybrid.to(device)

            output_hybrid = diff_simplex.forward_backward(
                    unet_hybrid, img.reshape(1, 1, *args_hybrid["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_hybrid["noise_fn"]
                    )

            unet_hybrid.cpu()

            mse_hybrid = (img - output_hybrid).square()
            hybrid_sqe.append(mse_hybrid.detach().cpu().numpy().flatten())

            fpr_hybrid, tpr_hybrid, threshold_hybrid = evaluation.ROC_AUC(img_mask, mse_hybrid)
            hybrid_AUC.append(evaluation.AUC_score(fpr_hybrid, tpr_hybrid))

            unet_gauss.to(device)
            output_gauss = diff_gauss.forward_backward(
                    unet_gauss, img.reshape(1, 1, *args_simplex["img_size"]),
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_gauss["noise_fn"]
                    )

            unet_gauss.cpu()

            mse_gauss = (img - output_gauss).square()
            gauss_sqe.append(mse_gauss.detach().cpu().numpy().flatten())
            fpr_gauss, tpr_gauss, threshold_gauss = evaluation.ROC_AUC(img_mask, mse_gauss)
            gauss_AUC.append(evaluation.AUC_score(fpr_gauss, tpr_gauss))

            img = img_128_whole[slice_number, ...].reshape(1, 1, *args_GAN["img_size"]).to(device)
            img_mask = img_mask_128_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args_GAN["img_size"])
            img_mask_center = img_mask[:, :,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_center = img[:, :, args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                         args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_128.append(img_mask_center.detach().cpu().numpy().flatten())
            input_cropped = input_cropped.to(device)
            netG.to(device)
            input_cropped.resize_(img.size()).copy_(img)
            # recon_image = input_cropped.clone()
            with torch.no_grad():
                input_cropped[:, 0,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize] \
                    = 0

            fake = netG(input_cropped)
            # print(fake.shape, img.shape, recon_image.shape)
            # recon_image.data[:, :, args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
            # args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4] = fake.data

            mse_GAN = (img_center - fake).square()
            GAN_sqe.append(mse_GAN.detach().cpu().numpy().flatten())
            fpr_GAN, tpr_GAN, threshold_GAN = evaluation.ROC_AUC(img_mask_center, mse_GAN)
            GAN_AUC.append(evaluation.AUC_score(fpr_GAN, tpr_GAN))

            input_cropped.cpu()
            netG.cpu()

            plt.plot(fpr_gauss, tpr_gauss, ":", label=f"gauss AUC={gauss_AUC[-1]:.2f}")
            plt.plot(fpr_simplex, tpr_simplex, "-", label=f"simplex AUC={simplex_AUC[-1]:.2f}")
            plt.plot(fpr_GAN, tpr_GAN, "-.", label=f"GAN AUC={GAN_AUC[-1]:.2f}")
            plt.legend()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])
            plt.savefig(
                    f'./metrics/ROC_data_3/{new_128["filenames"][0][-9:-4]}'
                    f'-{new_128["slices"][slice_number].cpu().item()}.png'
                    )
            plt.clf()

    simplex_sqe = np.array(simplex_sqe)
    gauss_sqe = np.array(gauss_sqe)
    GAN_sqe = np.array(GAN_sqe)
    hybrid_sqe = np.array(hybrid_sqe)
    img_256 = np.array(img_256)
    img_128 = np.array(img_128)

    fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(img_256, simplex_sqe)
    fpr_gauss, tpr_gauss, _ = evaluation.ROC_AUC(img_256, gauss_sqe)
    fpr_GAN, tpr_GAN, _ = evaluation.ROC_AUC(img_128, GAN_sqe)
    fpr_hybrid, tpr_hybrid, _ = evaluation.ROC_AUC(img_256, hybrid_sqe)

    for model in [(fpr_simplex, tpr_simplex, "simplex"), (fpr_gauss, tpr_gauss, "gauss"), (fpr_GAN, tpr_GAN, "GAN"),
                  (fpr_hybrid, tpr_hybrid, "hybrid")]:

        with open(f'./metrics/ROC_data_2/overall_{model[2]}.csv', mode="w") as f:
            f.write(f"fpr, tpr, {evaluation.AUC_score(model[0], model[1])}")
            f.write("\n")
            for i in range(len(model[0])):
                f.write(",".join([f"{j:.4f}" for j in [model[0][i], model[1][i]]]))
                f.write("\n")

    plt.plot(fpr_gauss, tpr_gauss, ":", label=f"Gaussian AUC={evaluation.AUC_score(fpr_gauss, tpr_gauss):.3f}")
    plt.plot(
            fpr_simplex, tpr_simplex, "-",
            label=f"Simplex $\mathcal{{L}}_{{simple}}$ AUC={evaluation.AUC_score(fpr_simplex, tpr_simplex):.3f}"
            )
    plt.plot(
            fpr_hybrid, tpr_hybrid, "-", label=f"Simplex $\mathcal{{L}}_{{hybrid}}$ AUC"
                                               f"={evaluation.AUC_score(fpr_hybrid, tpr_hybrid):.3f}"
            )
    plt.plot(
            fpr_GAN, tpr_GAN, "-.",
            label=f"Adversarial Context Encoder AUC={evaluation.AUC_score(fpr_GAN, tpr_GAN):.3f}"
            )
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f'./metrics/ROC_data_2/Overall.png')
    plt.clf()

    print(f"Simplex AUC {np.mean(simplex_AUC)} +- {np.std(simplex_AUC)}")
    print(f"Simplex hybrid AUC {np.mean(hybrid_AUC)} +- {np.std(hybrid_AUC)}")
    print(f"Gauss AUC {np.mean(gauss_AUC)} +- {np.std(gauss_AUC)}")
    print(f"CE AUC {np.mean(GAN_AUC)} +- {np.std(GAN_AUC)}")


def gan_anomalous():
    import Comparative_models.CE as CE
    args, output = load_parameters(device)
    args["Batch_Size"] = 1

    netG = CE.Generator(start_size=args['img_size'][0], out_size=args['inpaint_size'], dropout=args["dropout"])

    netG.load_state_dict(output["generator_state_dict"])
    netG.to(device)
    netG.eval()
    ano_dataset = dataset.AnomalousMRIDataset(
            ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
            slice_selection="iterateKnown_restricted", resized=False
            )

    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    overlapSize = args['overlap']
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256)
    input_cropped = input_cropped.to(device)

    try:
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous')
    except OSError:
        pass

    # make folder for each anomalous volume
    for i in ano_dataset.slices.keys():
        try:
            os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{i}')
        except OSError:
            pass

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        image = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])

        img_mask_whole = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            img = image[slice_number, ...].to(device).reshape(1, 1, *args["img_size"])
            img_mask = img_mask_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args["img_size"])

            if args['type'] == 'sliding':
                recon_image = ce_sliding_window(img, netG, input_cropped, args)
            else:
                input_cropped.resize_(img.size()).copy_(img)
                recon_image = input_cropped.clone()
                with torch.no_grad():
                    input_cropped[:, 0,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] \
                        = 0

                fake = netG(input_cropped)

                recon_image.data[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4] = fake.data

            mse = (img - recon_image).square()
            mse = (mse > 0.5).float()
            dice_data.append(
                    evaluation.dice_coeff(img, recon_image.to(device), img_mask, mse=mse).detach().cpu().numpy()
                    )
            ssim_data.append(

                    evaluation.SSIM(img.reshape(*args["img_size"]), recon_image.reshape(*args["img_size"]))
                    )
            precision.append(evaluation.precision(img_mask, mse).detach().cpu().numpy())
            recall.append(evaluation.recall(img_mask, mse).detach().cpu().numpy())
            IOU.append(evaluation.IoU(img_mask, mse))
            FPR.append(evaluation.FPR(img_mask, mse).detach().cpu().numpy())
            plt.close('all')

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +-{np.std(ssim_data[-4:])}")
        print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
        print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
        print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
        print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")
        print("\n")

    print()
    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    start_time = time.time()
    for i in range(len(ano_dataset)):

        new = next(loader)
        image = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])

        img_mask_whole = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        for slice_number in range(4):
            try:
                os.makedirs(
                        f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                        f'{new["slices"][slice_number].numpy()[0]}'
                        )
            except OSError:
                pass
            img = image[slice_number, ...].to(device).reshape(1, 1, *args["img_size"])
            img_mask = img_mask_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args["img_size"])

            img_center = img[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                         args['img_size'][0] // 4: args['inpaint_size'] + args['img_size'][0] // 4]
            img_mask_center = img_mask[:, :, args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                              args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4]

            input_cropped.resize_(img.size()).copy_(img)
            with torch.no_grad():
                input_cropped[:, 0,
                args['img_size'][0] // 4 + overlapSize:
                args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                args['img_size'][0] // 4 + overlapSize:
                args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] \
                    = 0

            fake = netG(input_cropped)

            mse = (img_center - fake).square()
            mse = (mse > 0.5).float()
            dice_data.append(
                    evaluation.dice_coeff(img_center, fake, img_mask_center, mse=mse).detach().cpu().numpy()
                    )
            ssim_data.append(
                    evaluation.SSIM(
                            img_center.reshape(args["inpaint_size"], args["inpaint_size"]),
                            fake.reshape(args["inpaint_size"], args["inpaint_size"])
                            )
                    )
            precision.append(evaluation.precision(img_mask_center, mse).detach().cpu().numpy())
            recall.append(evaluation.recall(img_mask_center, mse).detach().cpu().numpy())
            IOU.append(evaluation.IoU(img_mask_center, mse))
            FPR.append(evaluation.FPR(img_mask_center, mse).detach().cpu().numpy())
            plt.close('all')

        time_taken = time.time() - start_time
        remaining_epochs = 22 - i
        time_per_epoch = time_taken / (i + 1)
        hours = remaining_epochs * time_per_epoch / 3600
        mins = (hours % 1) * 60
        hours = int(hours)

        print(
                f"file: {new['filenames'][0][-9:-4]}, "
                f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining time: {hours}:{mins:02.0f}"
                )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +-{np.std(ssim_data[-4:])}")
        print(f"Dice: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(precision[-4:])} +- {np.std(precision[-4:])}")
        print(f"Recall: {np.mean(recall[-4:])} +- {np.std(recall[-4:])}")
        print(f"FPR: {np.mean(FPR[-4:])} +- {np.std(FPR[-4:])}")
        print(f"IOU: {np.mean(IOU[-4:])} +- {np.std(IOU[-4:])}")
        print("\n")

    print()
    print(f"Dice coefficient over all recorded segmentations: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(
            f"Structural Similarity Index (SSIM) over all recorded segmentations: {np.mean(ssim_data)} +-"
            f" {np.std(ssim_data)}"
            )
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")


def ce_sliding_window(img, netG, input_cropped, args):
    input_cropped.resize_(img.size()).copy_(img)
    recon_image = input_cropped.clone()
    for center_offset_y in np.arange(0, 97, args['inpaint_size']):

        for center_offset_x in np.arange(0, 97, args['inpaint_size']):
            with torch.no_grad():
                input_cropped.resize_(img.size()).copy_(img)
                input_cropped[:, 0,
                center_offset_x + args['overlap']: args['inpaint_size'] + center_offset_x - args['overlap'],
                center_offset_y + args['overlap']: args['inpaint_size'] + center_offset_y - args[
                    'overlap']] = 0

            fake = netG(input_cropped)

            recon_image.data[:, :, center_offset_x:args['inpaint_size'] + center_offset_x,
            center_offset_y:args['inpaint_size'] + center_offset_y] = fake.data

    return recon_image

if __name__ == "__main__":
    import sys
    from matplotlib import font_manager

    font_path = "./times.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     graph_data()
    anomalous_validation_1()
#     anomalous_metric_calculation()